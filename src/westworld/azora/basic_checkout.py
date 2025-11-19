import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, TypedDict
from zoneinfo import ZoneInfo

from pydantic import BaseModel

from westworld.base import (
    BaseMetric,
    BaseTaskConfig,
    UserMetadata,
    async_retry_with_exponential_backoff,
    get_import_path,
)


if TYPE_CHECKING:
    from playwright.async_api import Page


# =========================
# Core domain models (from verifier)
# =========================


@dataclass
class ContactInfo:
    """Contact information for the checkout."""

    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None


@dataclass
class AddressInfo:
    """Address information for the checkout."""

    id: Optional[str] = None
    full_name: Optional[str] = None
    street_address: Optional[str] = None
    apt_suite: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    country: Optional[str] = None
    phone_number: Optional[str] = None
    used_default: Optional[bool] = None
    is_pickup: Optional[bool] = None
    pickup_location: Optional[Dict[str, Any]] = None


@dataclass
class PaymentInfo:
    """Payment information for the checkout."""

    payment_method_type: Optional[str] = None
    payment_method_id: Optional[str] = None
    card_type: Optional[str] = None
    card_last_four: Optional[str] = None
    expiration_month: Optional[str] = None
    expiration_year: Optional[str] = None
    name_on_card: Optional[str] = None
    used_default: Optional[bool] = None
    billing_address_same: Optional[bool] = None
    billing_address: Optional[Dict[str, Any]] = None  # Full billing address details
    gift_card_code: Optional[str] = None
    gift_card_balance: Optional[float] = None
    is_default: Optional[bool] = None


@dataclass
class ShippingInfo:
    """Shipping information for the checkout."""

    selected_option: Optional[str] = None  # "free", "standard", "sameday"
    delivery_date: Optional[str] = None  # YYYY-MM-DD
    delivery_date_display: Optional[str] = None  # "Sunday, Aug 3"
    selected_time_slot: Optional[str] = None  # "Today 2 PM - 6 PM"
    estimated_cost: Optional[float] = None
    is_pickup: Optional[bool] = None


@dataclass
class CartInfo:
    """Cart information for the checkout."""

    items: List[Dict[str, Any]] = None
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    shipping_cost: Optional[float] = None
    total: Optional[float] = None
    total_items: Optional[int] = None
    is_gift: Optional[bool] = None


@dataclass
class ConfirmationInfo:
    """Confirmation information for the checkout."""

    reached: bool = False
    order_number: Optional[str] = None
    confirmation_text_present: bool = False
    delivery_date: Optional[str] = None
    delivery_address: Optional[str] = None
    payment_method: Optional[str] = None
    order_total: Optional[float] = None


@dataclass
class ApplicationState:
    """Current state of the application."""

    contact: ContactInfo
    address: AddressInfo
    payment: PaymentInfo
    shipping: ShippingInfo
    cart: CartInfo
    confirmation: ConfirmationInfo
    timeline: List[Dict[str, Any]]
    current_url: Optional[str] = None
    current_page: Optional[str] = None  # "cart", "checkout", "confirmation", etc.
    form_states: Optional[Dict[str, bool]] = None


@dataclass
class BasicCheckoutParams:
    """Expected checkout parameters from the basic checkout task."""

    # Extracted from task description
    delivery_recipient: Optional[str] = None  # Jennifer Morrison, Sharon Samples, etc.
    card_ending: Optional[str] = None  # 6685, 4931, 7792
    delivery_date_text: Optional[str] = None  # "Today", "three days from now", etc.
    delivery_date_option_id: Optional[str] = None  # "free", "standard", "sameday"

    # User info from task values
    title: Optional[str] = None
    given_name: Optional[str] = None
    middle_initial: Optional[str] = None
    surname: Optional[str] = None
    street_address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    state_full: Optional[str] = None
    zip_code: Optional[str] = None
    country: Optional[str] = None
    country_full: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    phone_country_code: Optional[str] = None
    cc_type: Optional[str] = None
    cc_number: Optional[str] = None
    cvv2: Optional[str] = None
    cc_expires: Optional[str] = None


@dataclass
class AgentOutput:
    """Optional data the agent claims to have read from the page."""

    other: Dict[str, Any]
    order_number: Optional[str] = None
    raw_response: Optional[str] = None


@dataclass
class VerificationResult:
    """Result of the verification process."""

    passed: bool
    details: Dict[str, Any]
    failures: List[str]


# =========================
# Verification helpers (from verifier)
# =========================


def load_card_ownership_mapping() -> Dict[str, str]:
    """Load card ownership mapping from people.json data.

    Returns:
        Dictionary mapping card endings to full names (e.g., {"6685": "Jennifer Morrison"})
    """
    people_file = Path("data/people.json")
    if not people_file.exists():
        # Fallback to hardcoded mapping if people.json not available
        return {"6685": "Jennifer Morrison", "4931": "Sharon Samples", "7792": "Michelle Feller"}

    try:
        with open(people_file, "r") as f:
            people = json.load(f)

        card_owners = {}
        for person in people:
            cc_number = person.get("CCNumber", "")
            if cc_number and len(cc_number) >= 4:
                card_ending = cc_number[-4:]
                given_name = person.get("GivenName", "")
                middle_initial = person.get("MiddleInitial", "")
                surname = person.get("Surname", "")

                # Construct full name (include middle initial if present)
                if middle_initial:
                    full_name = f"{given_name} {middle_initial} {surname}".strip()
                else:
                    full_name = f"{given_name} {surname}".strip()

                card_owners[card_ending] = full_name

        return card_owners
    except (json.JSONDecodeError, FileNotFoundError, KeyError):
        # Fallback to hardcoded mapping on any error
        return {"6685": "Jennifer Morrison", "4931": "Sharon Samples", "7792": "Michelle Feller"}


def normalize_string_for_comparison(text: str) -> str:
    """Normalize string for case-insensitive comparison."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip().lower()


def normalize_name_for_comparison(name: str) -> str:
    """Normalize name for comparison, ignoring middle initials.

    This function removes middle initials (single letters) from names
    to allow flexible matching. For example:
    - "Michelle D Feller" -> "michelle feller"
    - "Jennifer A Morrison" -> "jennifer morrison"
    - "Sharon Samples" -> "sharon samples"
    """
    if not name:
        return ""

    # Normalize whitespace and convert to lowercase
    normalized = re.sub(r"\s+", " ", name).strip().lower()

    # Split into words
    words = normalized.split()

    if len(words) <= 2:
        # If 1-2 words, return as is (no middle initial to remove)
        return normalized

    # Only filter out middle initials if we have at least 3 words and the middle word is a single letter
    if len(words) >= 3 and len(words[1]) == 1:
        # Remove the middle initial (second word)
        filtered_words = [words[0]] + words[2:]
        return " ".join(filtered_words)

    # Otherwise, return as is
    return normalized


def get_last4_digits(card_number: Optional[str]) -> str:
    """Extract last 4 digits from card number."""
    if not card_number:
        return ""
    digits = re.sub(r"\D", "", card_number)
    return digits[-4:] if len(digits) >= 4 else ""


def normalize_expiry(expiry: str) -> str:
    """Normalize expiry to canonical YYYY-MM string.

    Accepts formats like M/YY, MM/YY, M/YYYY, MM/YYYY.
    Returns empty string on invalid input.
    """
    if not expiry:
        return ""
    s = expiry.strip()
    m = re.match(r"^(\d{1,2})\s*[\/\-\s]\s*(\d{2,4})$", s)
    if m:
        month = int(m.group(1))
        year = int(m.group(2))
        if year < 100:
            year += 2000
        if 1 <= month <= 12:
            return f"{year:04d}-{month:02d}"
    return ""


def parse_delivery_date_text(date_text: str) -> str:
    """Convert delivery date text to actual date.

    Args:
        date_text: Text like "Today", "three days from now", etc.

    Returns:
        Date in YYYY-MM-DD format
    """
    today = datetime.now()
    date_text_lower = date_text.lower().strip()

    if date_text_lower == "today":
        delivery_date = today
    elif date_text_lower == "tomorrow":
        delivery_date = today + timedelta(days=1)
    elif "three days from now" in date_text_lower:
        delivery_date = today + timedelta(days=3)
    elif "five days from now" in date_text_lower:
        delivery_date = today + timedelta(days=5)
    else:
        # Default to today if unrecognized
        delivery_date = today

    return delivery_date.strftime("%Y-%m-%d")


def parse_date_string(date_str: str) -> Optional[datetime]:
    """Parse a date string in YYYY-MM-DD format to datetime object.

    Args:
        date_str: Date string in YYYY-MM-DD format

    Returns:
        datetime object or None if parsing fails
    """
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def dates_match(expected_date_text: str, observed_date_str: Optional[str]) -> bool:
    """Compare expected delivery date text with observed date string using actual dates.

    Args:
        expected_date_text: ISO date string (YYYY-MM-DD) from evaluation engine,
                          or text like "Today", "three days from now" for backward compatibility
        observed_date_str: Date string from simulator in YYYY-MM-DD format

    Returns:
        True if dates match, False otherwise
    """
    if not observed_date_str:
        return False

    # If expected_date_text is already in ISO format (YYYY-MM-DD), use it directly
    if isinstance(expected_date_text, str) and len(expected_date_text) == 10 and expected_date_text.count("-") == 2:
        expected_date_str = expected_date_text
    else:
        # Otherwise, parse the text format (backward compatibility)
        expected_date_str = parse_delivery_date_text(expected_date_text)

    expected_date = parse_date_string(expected_date_str)
    observed_date = parse_date_string(observed_date_str)

    if not expected_date or not observed_date:
        return False

    return expected_date.date() == observed_date.date()


def extract_recipient_from_task(task_description: str) -> Optional[str]:
    """Extract recipient name from task description.

    Args:
        task_description: The task description text

    Returns:
        Recipient name if found, None otherwise
    """
    task_lower = task_description.lower()

    if "jennifer morrison" in task_lower:
        return "Jennifer Morrison"
    elif "sharon samples" in task_lower:
        return "Sharon Samples"
    elif "michelle feller" in task_lower:
        return "Michelle Feller"

    return None


def extract_card_ending_from_task(task_description: str) -> Optional[str]:
    """Extract card ending digits from task description.

    Args:
        task_description: The task description text

    Returns:
        Card ending digits if found, None otherwise
    """
    task_lower = task_description.lower()

    if "ending in 6685" in task_lower:
        return "6685"
    elif "ending in 4931" in task_lower:
        return "4931"
    elif "ending in 7792" in task_lower:
        return "7792"

    return None


def extract_delivery_date_from_task(task_description: str) -> Optional[str]:
    """Extract delivery date from task description.

    Args:
        task_description: The task description text

    Returns:
        Delivery date text if found, None otherwise
    """
    task_lower = task_description.lower()

    if "today" in task_lower:
        return "Today"
    elif "three days from now" in task_lower:
        return "three days from now"
    elif "five days from now" in task_lower:
        return "five days from now"
    elif "tomorrow" in task_lower:
        return "tomorrow"

    return None


def verify_delivery_recipient(params: BasicCheckoutParams, observed: ApplicationState) -> Dict[str, Any]:
    """Verify delivery recipient matches expected name."""
    failures = []
    checks = []

    def add_check(name: str, expected: str, observed_val: str, passed: bool):
        checks.append({"check": name, "expected": expected, "observed": observed_val, "passed": passed})
        if not passed:
            failures.append(name)

    if params.delivery_recipient:
        expected_recipient = normalize_name_for_comparison(params.delivery_recipient)
        observed_recipient = normalize_name_for_comparison(observed.address.full_name or "")
        passed = expected_recipient == observed_recipient
        add_check("delivery_recipient_match", params.delivery_recipient, observed.address.full_name or "", passed)
    else:
        add_check("delivery_recipient_extraction", "recipient name from task", "none extracted", False)

    return {"passed": len(failures) == 0, "failures": failures, "checks": checks}


def verify_payment_method(params: BasicCheckoutParams, observed: ApplicationState) -> Dict[str, Any]:
    """Verify payment method matches expected card."""
    failures = []
    checks = []

    def add_check(name: str, expected: str, observed_val: str, passed: bool):
        checks.append({"check": name, "expected": expected, "observed": observed_val, "passed": passed})
        if not passed:
            failures.append(name)

    if params.card_ending:
        expected_ending = params.card_ending
        observed_ending = observed.payment.card_last_four or ""
        passed = expected_ending == observed_ending
        add_check("card_ending_match", expected_ending, observed_ending, passed)

        # Verify payment method type is credit card when using a specific card
        expected_type = "credit-card"
        observed_type = observed.payment.payment_method_type or ""
        passed_type = expected_type == observed_type
        add_check("payment_method_type", expected_type, observed_type, passed_type)
    else:
        add_check("card_ending_extraction", "card ending from task", "none extracted", False)

    return {"passed": len(failures) == 0, "failures": failures, "checks": checks}


def verify_delivery_date(params: BasicCheckoutParams, observed: ApplicationState) -> Dict[str, Any]:
    """Verify delivery date by comparing delivery_date_option_id instead of actual dates."""
    failures = []
    checks = []

    def add_check(name: str, expected: str, observed_val: str, passed: bool):
        checks.append({"check": name, "expected": expected, "observed": observed_val, "passed": passed})
        if not passed:
            failures.append(name)

    # Verify by delivery_date_option_id (e.g., "free", "standard", "sameday")
    if params.delivery_date_option_id:
        if observed.shipping.selected_option:
            # Compare option IDs directly
            passed = params.delivery_date_option_id == observed.shipping.selected_option
            add_check(
                "delivery_date_option_match", params.delivery_date_option_id, observed.shipping.selected_option, passed
            )
        else:
            # No delivery option info available
            add_check("delivery_date_option_missing", params.delivery_date_option_id, "None", False)
    else:
        add_check("delivery_date_option_id_missing", "delivery_date_option_id from task", "none found", False)

    return {"passed": len(failures) == 0, "failures": failures, "checks": checks}


def verify_billing_address(params: BasicCheckoutParams, observed: ApplicationState) -> Dict[str, Any]:
    """Verify billing address matches cardholder information."""
    failures = []
    checks = []

    def add_check(name: str, expected: str, observed_val: str, passed: bool):
        checks.append({"check": name, "expected": expected, "observed": observed_val, "passed": passed})
        if not passed:
            failures.append(name)

    if not params.card_ending:
        add_check("card_ending_available", "card ending from task", "none extracted", False)
        return {"passed": len(failures) == 0, "failures": failures, "checks": checks}

    # Load dynamic card ownership mapping from people.json
    card_owners = load_card_ownership_mapping()

    if params.card_ending in card_owners:
        # Card has a known owner - billing address should match card owner
        expected_cardholder = normalize_name_for_comparison(card_owners[params.card_ending])

        # Check if billing address is set correctly
        billing_same = observed.payment.billing_address_same
        if billing_same is True:
            # When billing same as delivery, delivery should match cardholder
            delivery_name = normalize_name_for_comparison(observed.address.full_name or "")
            passed = expected_cardholder == delivery_name
            add_check(
                "billing_same_delivery_match",
                f"delivery should match cardholder ({expected_cardholder}) when billing_address_same=True",
                delivery_name,
                passed,
            )
        elif billing_same is False:
            # When billing is separate, that's acceptable - we trust the simulator handled it correctly
            add_check(
                "billing_address_separate", "billing address separate from delivery", "billing_address_same=False", True
            )
        elif billing_same is None:
            # billing_address_same is None - this might happen if the simulator doesn't track this
            # We'll be lenient here and just log it, but not fail the test
            add_check("billing_address_info_available", "billing_address_same not tracked by simulator", "None", True)
        else:
            add_check(
                "billing_address_info_available", "billing_address_same should be determined", str(billing_same), False
            )

        # If we have user_info and it matches the cardholder, verify billing address details when available
        if (
            params.given_name
            and params.surname
            and normalize_name_for_comparison(f"{params.given_name} {params.surname}") == expected_cardholder
        ):
            # Check name on card if available
            if observed.payment.name_on_card:
                observed_name = normalize_name_for_comparison(observed.payment.name_on_card)
                passed = expected_cardholder == observed_name
                add_check("cardholder_name_match", expected_cardholder, observed_name, passed)

            # Verify billing address details if available
            if observed.payment.billing_address and params.street_address:
                billing_address = observed.payment.billing_address
                expected_street = normalize_string_for_comparison(params.street_address or "")
                expected_city = normalize_string_for_comparison(params.city or "")
                expected_state = normalize_string_for_comparison(params.state or "")
                expected_zip = normalize_string_for_comparison(params.zip_code or "")

                observed_street = normalize_string_for_comparison(billing_address.get("street_address", ""))
                observed_city = normalize_string_for_comparison(billing_address.get("city", ""))
                observed_state = normalize_string_for_comparison(billing_address.get("state", ""))
                observed_zip = normalize_string_for_comparison(billing_address.get("zip_code", ""))

                # Check each address component
                street_match = expected_street == observed_street
                city_match = expected_city == observed_city
                state_match = expected_state == observed_state
                zip_match = expected_zip == observed_zip

                add_check("billing_address_street_match", expected_street, observed_street, street_match)
                add_check("billing_address_city_match", expected_city, observed_city, city_match)
                add_check("billing_address_state_match", expected_state, observed_state, state_match)
                add_check("billing_address_zip_match", expected_zip, observed_zip, zip_match)
    else:
        # Card ending not found in people.json - this shouldn't happen with the new generator
        # but we'll be lenient and just log it
        add_check("card_ending_recognized", f"card ending {params.card_ending} found in people.json", "not found", True)

    return {"passed": len(failures) == 0, "failures": failures, "checks": checks}


def verify_confirmation(
    params: BasicCheckoutParams, observed: ApplicationState, agent_output: Optional[AgentOutput]
) -> Dict[str, Any]:
    """Verify order confirmation was reached."""
    failures = []
    checks = []

    def add_check(name: str, expected: str, observed_val: str, passed: bool):
        checks.append({"check": name, "expected": expected, "observed": observed_val, "passed": passed})
        if not passed:
            failures.append(name)

    # Check if confirmation page was reached
    confirmation_reached = observed.confirmation.reached or (observed.current_page == "confirmation")
    add_check("confirmation_page_reached", "true", str(confirmation_reached), confirmation_reached)

    # Check if order number is available (either from simulator state or agent)
    order_number = observed.confirmation.order_number or (agent_output.order_number if agent_output else None)
    order_number_present = bool(order_number)
    add_check("order_number_present", "true", str(order_number_present), order_number_present)

    # Check confirmation text presence
    if observed.confirmation.confirmation_text_present is not None:
        add_check(
            "confirmation_text_present",
            "true",
            str(observed.confirmation.confirmation_text_present),
            observed.confirmation.confirmation_text_present,
        )

    return {"passed": len(failures) == 0, "failures": failures, "checks": checks}


def create_application_state_from_dict(data: Dict[str, Any]) -> ApplicationState:
    """Create ApplicationState from dictionary data from the simulator."""
    contact = ContactInfo(
        first_name=data.get("contact", {}).get("first_name"),
        last_name=data.get("contact", {}).get("last_name"),
        email=data.get("contact", {}).get("email"),
        phone=data.get("contact", {}).get("phone"),
    )

    address = AddressInfo(
        id=data.get("address", {}).get("id"),
        full_name=data.get("address", {}).get("full_name"),
        street_address=data.get("address", {}).get("street_address"),
        apt_suite=data.get("address", {}).get("apt_suite"),
        city=data.get("address", {}).get("city"),
        state=data.get("address", {}).get("state"),
        zip_code=data.get("address", {}).get("zip_code"),
        country=data.get("address", {}).get("country"),
        phone_number=data.get("address", {}).get("phone_number"),
        used_default=data.get("address", {}).get("used_default"),
        is_pickup=data.get("address", {}).get("is_pickup"),
        pickup_location=data.get("address", {}).get("pickup_location"),
    )

    payment = PaymentInfo(
        payment_method_type=data.get("payment", {}).get("payment_method_type"),
        payment_method_id=data.get("payment", {}).get("payment_method_id"),
        card_type=data.get("payment", {}).get("card_type"),
        card_last_four=data.get("payment", {}).get("card_last_four"),
        expiration_month=data.get("payment", {}).get("expiration_month"),
        expiration_year=data.get("payment", {}).get("expiration_year"),
        name_on_card=data.get("payment", {}).get("name_on_card"),
        used_default=data.get("payment", {}).get("used_default"),
        billing_address_same=data.get("payment", {}).get("billing_address_same"),
        billing_address=data.get("payment", {}).get("billing_address"),
        gift_card_code=data.get("payment", {}).get("gift_card_code"),
        gift_card_balance=data.get("payment", {}).get("gift_card_balance"),
        is_default=data.get("payment", {}).get("is_default"),
    )

    shipping = ShippingInfo(
        selected_option=data.get("shipping", {}).get("selected_option"),
        delivery_date=data.get("shipping", {}).get("delivery_date"),
        delivery_date_display=data.get("shipping", {}).get("delivery_date_display"),
        selected_time_slot=data.get("shipping", {}).get("selected_time_slot"),
        estimated_cost=data.get("shipping", {}).get("estimated_cost"),
        is_pickup=data.get("shipping", {}).get("is_pickup"),
    )

    cart = CartInfo(
        items=data.get("cart", {}).get("items", []),
        subtotal=data.get("cart", {}).get("subtotal"),
        tax=data.get("cart", {}).get("tax"),
        shipping_cost=data.get("cart", {}).get("shipping_cost"),
        total=data.get("cart", {}).get("total"),
        total_items=data.get("cart", {}).get("total_items"),
        is_gift=data.get("cart", {}).get("is_gift"),
    )

    confirmation = ConfirmationInfo(
        reached=data.get("confirmation", {}).get("reached", False),
        order_number=data.get("confirmation", {}).get("order_number"),
        confirmation_text_present=data.get("confirmation", {}).get("confirmation_text_present", False),
        delivery_date=data.get("confirmation", {}).get("delivery_date"),
        delivery_address=data.get("confirmation", {}).get("delivery_address"),
        payment_method=data.get("confirmation", {}).get("payment_method"),
        order_total=data.get("confirmation", {}).get("order_total"),
    )

    return ApplicationState(
        contact=contact,
        address=address,
        payment=payment,
        shipping=shipping,
        cart=cart,
        confirmation=confirmation,
        timeline=data.get("timeline", []),
        current_url=data.get("current_url"),
        current_page=data.get("current_page"),
        form_states=data.get("form_states"),
    )


def create_basic_checkout_params_from_task(task_data: Dict[str, Any]) -> BasicCheckoutParams:
    """Create BasicCheckoutParams from task data."""
    task_description = task_data.get("task", "")
    values = task_data.get("values", {})
    user_info = values.get("user_info", {})

    # Prefer explicit ground truth values if available, otherwise extract from task description
    delivery_recipient = values.get("delivery_recipient") or extract_recipient_from_task(task_description)
    card_ending = values.get("card_ending") or extract_card_ending_from_task(task_description)

    # For delivery date, prefer evaluated ISO date from evaluation engine
    # If we have delivery_date_iso, use that to create the expected date text
    # Otherwise fall back to extracting from task description
    delivery_date_iso = values.get("delivery_date_iso")
    if delivery_date_iso:
        # Convert ISO date to the description format for comparison
        # The evaluation engine already evaluated the formula, so we use the ISO date directly
        delivery_date_text = delivery_date_iso  # Use ISO format for date matching
    else:
        # Fallback to old behavior: extract from task description
        delivery_date_text = values.get("delivery_date") or extract_delivery_date_from_task(task_description)

    # Get delivery_date_option_id for verification by option ID
    delivery_date_option_id = values.get("delivery_date_option_id")

    # If we have user_info with name details, construct the full name for comparison
    if delivery_recipient and user_info.get("GivenName") and user_info.get("Surname"):
        # Use the constructed name from user_info instead of the extracted one
        # This handles cases where task says "Michelle Feller" but user_info has "Michelle D Feller"
        given_name = user_info.get("GivenName", "")
        middle_initial = user_info.get("MiddleInitial", "")
        surname = user_info.get("Surname", "")

        constructed = f"{given_name} {middle_initial + ' ' if middle_initial else ''}{surname}".strip()

        # Compare ignoring single-letter middle initials
        same_person = normalize_name_for_comparison(delivery_recipient) == normalize_name_for_comparison(
            f"{given_name} {surname}"
        )

        # Only override when the task's recipient equals user_info person (ignoring MI)
        if same_person:
            delivery_recipient = constructed

    return BasicCheckoutParams(
        delivery_recipient=delivery_recipient,
        card_ending=card_ending,
        delivery_date_text=delivery_date_text,
        delivery_date_option_id=delivery_date_option_id,
        title=user_info.get("Title"),
        given_name=user_info.get("GivenName"),
        middle_initial=user_info.get("MiddleInitial"),
        surname=user_info.get("Surname"),
        street_address=user_info.get("StreetAddress"),
        city=user_info.get("City"),
        state=user_info.get("State"),
        state_full=user_info.get("StateFull"),
        zip_code=user_info.get("ZipCode"),
        country=user_info.get("Country"),
        country_full=user_info.get("CountryFull"),
        email=user_info.get("EmailAddress"),
        phone=user_info.get("TelephoneNumber"),
        phone_country_code=user_info.get("TelephoneCountryCode"),
        cc_type=user_info.get("CCType"),
        cc_number=user_info.get("CCNumber"),
        cvv2=user_info.get("CVV2"),
        cc_expires=user_info.get("CCExpires"),
    )


def verify_basic_checkout_complete(
    params: BasicCheckoutParams, observed: ApplicationState, agent_output: Optional[AgentOutput] = None
) -> VerificationResult:
    """Verify that the basic checkout task was completed successfully."""
    all_failures = []
    details = {}

    # Verify delivery recipient
    recipient_result = verify_delivery_recipient(params, observed)
    details["delivery_recipient"] = recipient_result
    all_failures.extend(recipient_result["failures"])

    # Verify payment method
    payment_result = verify_payment_method(params, observed)
    details["payment_method"] = payment_result
    all_failures.extend(payment_result["failures"])

    # Verify delivery date
    delivery_date_result = verify_delivery_date(params, observed)
    details["delivery_date"] = delivery_date_result
    all_failures.extend(delivery_date_result["failures"])

    # Verify billing address for new cards
    billing_address_result = verify_billing_address(params, observed)
    details["billing_address"] = billing_address_result
    all_failures.extend(billing_address_result["failures"])

    # Verify confirmation
    confirmation_result = verify_confirmation(params, observed, agent_output)
    details["confirmation"] = confirmation_result
    all_failures.extend(confirmation_result["failures"])

    # Add timeline for debugging
    details["timeline"] = observed.timeline

    passed = len(all_failures) == 0
    return VerificationResult(passed=passed, details=details, failures=all_failures)


# =========================
# Evaluator
# =========================


class InputDict(TypedDict, total=False):
    page: "Page"


class FinalResult(BaseModel):
    """Result model for basic checkout task."""

    score: float
    reasoning: str
    current_state: dict[str, Any]
    ground_truth: BasicCheckoutParams


class HalluminateAzoraBasicCheckout(BaseMetric):
    """Evaluator that runs verifier for Azora basic checkout task."""

    def __init__(self, task: str, values: dict[str, Any], user_metadata: UserMetadata) -> None:
        super().__init__()
        self.task = task
        self.values = values
        self.user_metadata = user_metadata
        # Create ground truth from task values (verifier expects {"task": task, "values": values})
        # The values should include delivery_date_iso if available (from _render_values)
        self._ground_truth = create_basic_checkout_params_from_task({"task": task, "values": values})
        self._page: "Page" | None = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(task={self.task}, values={self.values}, user_metadata={self.user_metadata})"

    async def reset(self) -> None:
        """Reset the evaluator state."""
        self._page = None

    async def update(self, /, **kwargs):
        inputs: InputDict = kwargs
        self._page = inputs.get("page")

    async def compute(self) -> FinalResult:
        if not self._page:
            return FinalResult(
                score=0.0,
                reasoning="Failed (environment): No page provided",
                current_state={},
                ground_truth=self._ground_truth,
            )

        current_state = await self._get_current_state()
        if not current_state:
            return FinalResult(
                score=0.0,
                reasoning="Failed (environment): Could not get current state",
                current_state={},
                ground_truth=self._ground_truth,
            )

        state_obj = create_application_state_from_dict(current_state)
        result = verify_basic_checkout_complete(self._ground_truth, state_obj)

        if result.passed:
            return FinalResult(
                score=1.0,
                reasoning="Passed all checks",
                current_state=current_state,
                ground_truth=self._ground_truth,
            )
        else:
            # Build detailed failure message with expected vs observed
            failure_details = []
            failure_details.append("Failed checks:")

            # Add details from each verification section
            # result.details is a dict where most values are dicts with "passed", "failures", "checks"
            # but "timeline" is a list, so we skip non-dict values
            for section_name, section_result in result.details.items():
                # Skip non-dict values (like "timeline" which is a list)
                if not isinstance(section_result, dict):
                    continue
                if not section_result.get("passed", True):
                    checks = section_result.get("checks", [])
                    for check in checks:
                        if isinstance(check, dict) and not check.get("passed", True):
                            expected = check.get("expected", "N/A")
                            observed = check.get("observed", "N/A")
                            check_name = check.get("check", "unknown")
                            failure_details.append(f"  {check_name}: expected '{expected}', observed '{observed}'")

            reasoning = "Failed (verification): " + "; ".join(result.failures)
            if len(failure_details) > 1:  # More than just the header
                reasoning += "\n" + "\n".join(failure_details)

            return FinalResult(
                score=0.0,
                reasoning=reasoning,
                current_state=current_state,
                ground_truth=self._ground_truth,
            )

    @async_retry_with_exponential_backoff()
    async def _get_current_state(self) -> dict[str, Any]:
        return await self._page.evaluate("() => window.__AMAZON_EVAL_STATE__")


def _evaluate_date_formula(formula_str: str, timezone: str, timestamp: int) -> datetime:
    """Safely evaluate a date formula string and return datetime object.

    Example:
        Input:  formula_str="now() + timedelta(days=5)", timezone="America/Los_Angeles", timestamp=1715702400
        Output: datetime object
    """

    def now():
        return datetime.fromtimestamp(timestamp, tz=ZoneInfo(timezone))

    safe_globals = {"timedelta": timedelta, "now": now}
    result = eval(formula_str, safe_globals)
    if not isinstance(result, datetime):
        raise ValueError(f"Formula did not return a datetime: {formula_str}")
    return result


def _format_date_for_task(date_obj: datetime) -> str:
    """Format datetime to match frontend's date format.

    Frontend uses: date.toLocaleDateString("en-US", { weekday: "long", month: "short", day: "numeric" })
    Example: "Thursday, Oct 23"
    """
    return date_obj.strftime("%A, %b %d")


def _render_values(values: dict[str, Any], timezone: str, timestamp: int) -> dict[str, Any]:
    """Render delivery_date_formula for basic_checkout tasks.

    Evaluates delivery_date_formula to get actual datetime, formats it for task rendering,
    and keeps delivery_date_description for verifier compatibility.

    IMPORTANT: The verifier's create_basic_checkout_params_from_task expects
    values.get("delivery_date") to be the TEXT format (e.g., "Today", "5 days from now"),
    which is stored in delivery_date_description. So we:
    1. Evaluate formula to get datetime → format for task → store in delivery_date
    2. Keep delivery_date_description for verifier (it will be used via delivery_date field)
    3. Store ISO date in delivery_date_iso for reference
    """
    evaluated = dict(values)

    # If delivery_date_formula exists, evaluate it
    if "delivery_date_formula" in evaluated:
        formula = evaluated["delivery_date_formula"]
        if isinstance(formula, str) and ("now()" in formula or "timedelta" in formula):
            try:
                # Evaluate: "now() + timedelta(days=5)" → datetime object
                evaluated_datetime = _evaluate_date_formula(formula, timezone, timestamp)

                # Format the date to match frontend format (e.g., "Thursday, Oct 23")
                formatted_date = _format_date_for_task(evaluated_datetime)

                # Store formatted date for task rendering
                evaluated["delivery_date"] = formatted_date

                # Store ISO date for reference
                evaluated["delivery_date_iso"] = evaluated_datetime.strftime("%Y-%m-%d")

                # The verifier expects delivery_date to be text format (e.g., "5 days from now")
                # So we need to ensure delivery_date_description is available
                # The verifier will extract it from the task text or use delivery_date_description
                if "delivery_date_description" in evaluated:
                    # Keep delivery_date_description for verifier compatibility
                    # The verifier will use this when extracting from task text
                    pass
                else:
                    # If no description, try to use the formatted date as fallback
                    # But verifier expects text like "5 days from now", not "Thursday, Oct 23"
                    pass

            except (ValueError, SyntaxError) as e:
                # If formula evaluation fails, keep original values
                print(f"Warning: Failed to evaluate delivery_date_formula '{formula}': {e}")
                # Fallback: use delivery_date_description if available
                if "delivery_date_description" in evaluated:
                    evaluated["delivery_date"] = evaluated["delivery_date_description"]

    return evaluated


def _render_task(task: str, values: dict[str, Any]) -> str:
    """Replace {delivery_recipient}, {card_ending}, {delivery_date}, {user_info} placeholders.

    Example:
        Input task: "Ship them to {delivery_recipient}. Use the card ending in {card_ending}..."
        Input values: {"delivery_recipient": "Jennifer Morrison", "card_ending": "6685",
                       "delivery_date": "Thursday, Oct 23", ...}
        Output: "Ship them to Jennifer Morrison. Use the card ending in 6685..."
    """
    result = task

    # Replace {delivery_recipient}
    if "{delivery_recipient}" in result:
        delivery_recipient = values.get("delivery_recipient", "")
        result = result.replace("{delivery_recipient}", delivery_recipient)

    # Replace {card_ending}
    if "{card_ending}" in result:
        card_ending = values.get("card_ending", "")
        result = result.replace("{card_ending}", card_ending)

    # Replace {delivery_date} with formatted date
    if "{delivery_date}" in result:
        # Use delivery_date (already formatted by _render_values) or fallback to description
        date_str = values.get("delivery_date") or values.get("delivery_date_description", "")
        if date_str:
            result = result.replace("{delivery_date}", date_str)

    # Replace {user_info} with string representation
    if "{user_info}" in result:
        user_info = values.get("user_info", {})
        result = result.replace("{user_info}", str(user_info))

    return result


def generate_task_config(
    url: str, task: str, values: dict[str, Any], location: str, timezone: str, timestamp: int | None = None
) -> BaseTaskConfig:
    """Generate a task config from raw parameters.

    This function:
    1. Replaces API key in URL
    2. Gets timestamp in user's timezone
    3. Evaluates date formulas in values
    4. Renders task text with formatted values
    5. Creates eval_config pointing to evaluator

    Args:
        url: Base URL with {HALLUMINATE_API_KEY} placeholder
        task: Task template with placeholders
        values: Task values including delivery_date_formula
        location: User location string
        timezone: User timezone (e.g., "America/Los_Angeles")
        timestamp: Optional timestamp (uses current time if None)

    Returns:
        BaseTaskConfig with rendered task, URL, user_metadata, and eval_config
    """
    # Replace the API key in the URL
    url = url.replace("{HALLUMINATE_API_KEY}", os.getenv("HALLUMINATE_API_KEY"))

    # Get the task start timestamp
    timestamp = int(datetime.now().timestamp()) if timestamp is None else timestamp

    # Create the user metadata
    user_metadata = UserMetadata(timestamp=timestamp, location=location, timezone=timezone)

    # Render the values (evaluate delivery_date_formula → actual date)
    values = _render_values(values, timezone, timestamp)

    # Render the task (replace placeholders with formatted values)
    task = _render_task(task, values)

    # Create the eval config
    eval_target = get_import_path(HalluminateAzoraBasicCheckout)
    eval_config = {
        "_target_": eval_target,
        "task": task,
        "values": values,
        "user_metadata": user_metadata.model_dump(),
    }

    # Create the task config
    return BaseTaskConfig(
        task=task,
        url=url,
        user_metadata=user_metadata,
        eval_config=eval_config,
    )


if __name__ == "__main__":
    from westworld.base import DatasetItem, instantiate

    dataset_row = {
        "task_id": "halluminate_bench/azora/basic_checkout/1",
        "task_generation_config_json": json.dumps(
            {
                "_target_": "halluminate_bench.azora.basic_checkout.generate_task_config",
                "url": "https://azora.halluminate.ai/?apiKey={HALLUMINATE_API_KEY}",
                "task": (
                    "Hey, can you order these AirPods for me? Ship them to {delivery_recipient}. "
                    "Use the card ending in {card_ending}. If you need to add a new card, make sure "
                    "the billing address matches the card owners address. Also, set the delivery date "
                    "to {delivery_date}. Let me know once the order is placed. User info: {user_info}"
                ),
                "values": {
                    "delivery_recipient": "Jennifer Morrison",
                    "card_ending": "6685",
                    "delivery_date_formula": "now() + timedelta(days=0)",
                    "delivery_date_text": "Today",
                    "user_info": {
                        "Title": "Mrs.",
                        "GivenName": "Michelle",
                        "MiddleInitial": "D",
                        "Surname": "Feller",
                        "StreetAddress": "1234 Ripple Street",
                        "City": "Saginaw",
                        "State": "MI",
                        "StateFull": "Michigan",
                        "ZipCode": "48607",
                        "Country": "US",
                        "CountryFull": "United States",
                        "EmailAddress": "MichelleDFeller@rhyta.com",
                        "TelephoneNumber": "989-730-9533",
                        "TelephoneCountryCode": "1",
                        "CCType": "MasterCard",
                        "CCNumber": "5114527275417792",
                        "CVV2": "640",
                        "CCExpires": "7/2028",
                    },
                },
                "location": "San Francisco, CA, United States",
                "timezone": "America/Los_Angeles",
                "timestamp": None,
            }
        ),
        "env": "sim",
        "domain": "azora",
        "l1_category": "e_commerce",
        "l2_category": "basic_checkout",
    }

    dataset_item = DatasetItem.model_validate(dataset_row)
    task_config = dataset_item.generate_task_config()
    evaluator = instantiate(task_config.eval_config)

    print("Loaded dataset item")
    print("-------------------")
    print(dataset_item)
    print()
    print("Generated task config")
    print("---------------------")
    print(task_config)
    print()
    print("Instantiated evaluator")
    print("----------------------")
    print(evaluator)
