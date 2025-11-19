import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
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
# Core domain models (copied from verifier)
# =========================


@dataclass
class ContactInfo:
    """Contact information for the delivery."""

    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None


@dataclass
class AddressInfo:
    """Address information for the delivery."""

    street_address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    country: Optional[str] = None


@dataclass
class PaymentInfo:
    """Payment information for the delivery."""

    payment_method: Optional[str] = None
    card_type: Optional[str] = None
    card_number: Optional[str] = None
    card_number_masked: Optional[str] = None
    expiry: Optional[str] = None
    used_default: Optional[bool] = None


@dataclass
class DeliveryInstructionInfo:
    """Delivery instruction information for the delivery."""

    recipient_name: Optional[str] = None
    delivery_date_text: Optional[str] = None  # "tomorrow", "2-days from now", etc.
    delivery_date: Optional[str] = None  # actual date in YYYY-MM-DD format
    delivery_method: Optional[str] = None
    property_type: Optional[str] = None  # "House", "Apartment", "Business", "Other"
    dropoff_location: Optional[str] = None  # "Front door", "Back door", "Building reception", etc.
    used_default_dropoff: Optional[bool] = None  # Track if user used default dropoff location


@dataclass
class ConfirmationInfo:
    """Confirmation information for the delivery instruction."""

    reached: bool = False
    order_number: Optional[str] = None
    confirmation_text_present: bool = False


@dataclass
class ApplicationState:
    """Current state of the delivery instruction application."""

    contact: ContactInfo
    address: AddressInfo
    payment: PaymentInfo
    delivery: DeliveryInstructionInfo
    confirmation: ConfirmationInfo
    timeline: List[Dict[str, Any]]
    url: Optional[str] = None
    default_address: Optional[AddressInfo] = None  # Default address from simulator
    default_payment: Optional[PaymentInfo] = None  # Default payment from simulator


@dataclass
class BasicDeliveryInstructionParams:
    """Expected delivery instruction parameters from the basic delivery instruction task."""

    delivery_address_type: str  # "default" or "user" or person name
    payment_method_type: str  # "default" or "user"
    property_type: str  # "House", "Apartment", "Business", "Other"
    delivery_location: str  # "Front door", "Back door", etc.
    delivery_date_text: Optional[str] = None  # "tomorrow", "2 days from now", etc. (for backward compatibility)
    delivery_date_iso: Optional[str] = None  # ISO date (YYYY-MM-DD) from evaluation engine (preferred)
    expected_person_name: Optional[str] = None  # Full name when using user-specific values
    # User info (same as in task values)
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
# Verification helpers (copied from verifier)
# =========================


def normalize_string_for_comparison(text: str) -> str:
    """Normalize string for case-insensitive comparison."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip().lower()


def normalize_expiry(expiry: str) -> str:
    """Normalize expiry to canonical YYYY-MM string.

    Accepts formats like M/YY, MM/YY, M/YYYY, MM/YYYY, M-YYYY, MM YYYY, and compact MMYYYY.
    Returns empty string on invalid input.
    """
    if not expiry:
        return ""
    s = expiry.strip()
    # With separator (/, -, or space)
    m = re.match(r"^(\d{1,2})\s*[\/\-\s]\s*(\d{2,4})$", s)
    if not m:
        # Compact MMYYYY (e.g., 072028)
        m = re.match(r"^(\d{1,2})(\d{4})$", s)
    if m:
        month = int(m.group(1))
        year = int(m.group(2))
        if year < 100:
            year += 2000
        if 1 <= month <= 12:
            return f"{year:04d}-{month:02d}"
        return ""
    return ""


def parse_delivery_date_text(date_text: str) -> str:
    """Convert delivery date text to actual date.

    This mirrors the simulator's parseDeliveryDateText function in eval.ts.

    Args:
        date_text: Text like "tomorrow", "2 days from now", etc.

    Returns:
        Date in YYYY-MM-DD format
    """
    today = datetime.now()
    date_text_lower = date_text.lower().strip()

    if date_text_lower == "tomorrow":
        delivery_date = today + timedelta(days=1)
    elif "2 days from now" in date_text_lower:
        delivery_date = today + timedelta(days=2)
    elif "3 days from now" in date_text_lower:
        delivery_date = today + timedelta(days=3)
    elif "4 days from now" in date_text_lower:
        delivery_date = today + timedelta(days=4)
    # Legacy format support for old tasks
    elif "2-days from now" in date_text_lower:
        delivery_date = today + timedelta(days=2)
    elif "3-days from now" in date_text_lower:
        delivery_date = today + timedelta(days=3)
    elif "4-days from now" in date_text_lower:
        delivery_date = today + timedelta(days=4)
    else:
        # Default to tomorrow if unrecognized
        delivery_date = today + timedelta(days=1)

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


def dates_match(expected_date: str, observed_date_str: Optional[str]) -> bool:
    """Compare expected delivery date with observed date string using actual dates.

    Args:
        expected_date: ISO date string (YYYY-MM-DD) from evaluation engine,
                      or text like "tomorrow", "2 days from now" for backward compatibility
        observed_date_str: Date string from simulator in YYYY-MM-DD format

    Returns:
        True if dates match, False otherwise
    """
    if not observed_date_str:
        return False

    # If expected_date is already in ISO format (YYYY-MM-DD), use it directly
    if isinstance(expected_date, str) and len(expected_date) == 10 and expected_date.count("-") == 2:
        expected_date_str = expected_date
    else:
        # Otherwise, parse the text format (backward compatibility)
        expected_date_str = parse_delivery_date_text(expected_date)

    expected_date_obj = parse_date_string(expected_date_str)
    observed_date_obj = parse_date_string(observed_date_str)

    if not expected_date_obj or not observed_date_obj:
        return False

    return expected_date_obj.date() == observed_date_obj.date()


def verify_contact_info(task: BasicDeliveryInstructionParams, observed: ApplicationState) -> Dict[str, Any]:
    """Verify contact information matches expected values."""
    failures = []
    checks = []

    def add_check(name: str, expected: str, observed_val: str, passed: bool):
        checks.append({"check": name, "expected": expected, "observed": observed_val, "passed": passed})
        if not passed:
            failures.append(name)

    # Only verify contact info when using user-specific delivery address
    if task.delivery_address_type == "user":
        if task.given_name:
            expected = normalize_string_for_comparison(task.given_name)
            observed_val = normalize_string_for_comparison(observed.contact.first_name)
            passed = expected == observed_val
            add_check("first_name_match", task.given_name, observed.contact.first_name, passed)

        if task.surname:
            expected = normalize_string_for_comparison(task.surname)
            observed_val = normalize_string_for_comparison(observed.contact.last_name)
            passed = expected == observed_val
            add_check("last_name_match", task.surname, observed.contact.last_name, passed)

        if task.phone:
            expected = re.sub(r"\D", "", task.phone)
            observed_val = re.sub(r"\D", "", observed.contact.phone or "")
            passed = normalize_string_for_comparison(expected) == normalize_string_for_comparison(observed_val)
            add_check("phone_match", expected, observed_val, passed)
    else:
        # For default delivery, we don't verify specific contact info
        add_check("contact_info_type", "default (no specific verification required)", "default", True)

    return {"passed": len(failures) == 0, "failures": failures, "checks": checks}


def verify_address_info(task: BasicDeliveryInstructionParams, observed: ApplicationState) -> Dict[str, Any]:
    """Verify address information matches expected values."""
    failures = []
    checks = []

    def add_check(name: str, expected: str, observed_val: str, passed: bool):
        checks.append({"check": name, "expected": expected, "observed": observed_val, "passed": passed})
        if not passed:
            failures.append(name)

    # Check if task requires default address and verify user actually used default
    if task.delivery_address_type == "default":
        # Verify that actual address matches default address
        if not observed.default_address:
            add_check(
                "default_address_missing",
                "default_shipping_info present in state",
                "None",
                False,
            )
        else:
            # Compare address fields with defaults
            expected = normalize_string_for_comparison(observed.default_address.street_address or "")
            observed_val = normalize_string_for_comparison(observed.address.street_address or "")
            passed = expected == observed_val
            add_check(
                "default_street_match",
                observed.default_address.street_address or "(empty)",
                observed.address.street_address or "(empty)",
                passed,
            )

            expected = normalize_string_for_comparison(observed.default_address.city or "")
            observed_val = normalize_string_for_comparison(observed.address.city or "")
            passed = expected == observed_val
            add_check(
                "default_city_match",
                observed.default_address.city or "(empty)",
                observed.address.city or "(empty)",
                passed,
            )

            expected = normalize_string_for_comparison(observed.default_address.state or "")
            observed_val = normalize_string_for_comparison(observed.address.state or "")
            passed = expected == observed_val
            add_check(
                "default_state_match",
                observed.default_address.state or "(empty)",
                observed.address.state or "(empty)",
                passed,
            )

            expected = normalize_string_for_comparison(observed.default_address.zip_code or "")[:5]
            observed_val = normalize_string_for_comparison(observed.address.zip_code or "")[:5]
            passed = expected == observed_val
            add_check(
                "default_zip_match",
                observed.default_address.zip_code or "(empty)",
                observed.address.zip_code or "(empty)",
                passed,
            )
    else:
        # For user-specific delivery address, verify against user_info
        if task.street_address:
            expected = normalize_string_for_comparison(task.street_address)
            observed_val = normalize_string_for_comparison(observed.address.street_address)
            passed = expected == observed_val
            add_check("delivery_street_match", task.street_address, observed.address.street_address, passed)

        if task.city:
            expected = normalize_string_for_comparison(task.city)
            observed_val = normalize_string_for_comparison(observed.address.city)
            passed = expected == observed_val
            add_check("delivery_city_match", task.city, observed.address.city, passed)

        task_state_norm = normalize_string_for_comparison(task.state) or normalize_string_for_comparison(
            task.state_full
        )
        observed_state_norm = normalize_string_for_comparison(observed.address.state)
        if task_state_norm:
            passed = task_state_norm == observed_state_norm
            add_check("delivery_state_match", task.state or task.state_full, observed.address.state, passed)

        if task.zip_code:
            expected = normalize_string_for_comparison(task.zip_code)[:5]
            observed_val = normalize_string_for_comparison(observed.address.zip_code)[:5]
            passed = expected == observed_val
            add_check("delivery_zip_match", task.zip_code, observed.address.zip_code, passed)

    return {"passed": len(failures) == 0, "failures": failures, "checks": checks}


def verify_payment_info(task: BasicDeliveryInstructionParams, observed: ApplicationState) -> Dict[str, Any]:
    """Verify payment information matches expected values."""
    failures = []
    checks = []

    def add_check(name: str, expected: str, observed_val: str, passed: bool):
        checks.append({"check": name, "expected": expected, "observed": observed_val, "passed": passed})
        if not passed:
            failures.append(name)

    # Check payment method type based on task values
    if task.payment_method_type == "default":
        # Verify that actual payment matches default payment
        if not observed.default_payment:
            add_check(
                "default_payment_missing",
                "default_payment_info present in state",
                "None",
                False,
            )
        else:
            # Compare payment fields with defaults
            expected = re.sub(r"\D", "", observed.default_payment.card_number or "")
            observed_val = re.sub(r"\D", "", observed.payment.card_number or "")
            passed = expected == observed_val
            add_check(
                "default_card_number_match",
                observed.default_payment.card_number or "(empty)",
                observed.payment.card_number or "(empty)",
                passed,
            )

            expected = normalize_expiry(observed.default_payment.expiry or "")
            observed_val = normalize_expiry(observed.payment.expiry or "")
            passed = expected == observed_val
            add_check(
                "default_expiry_match",
                observed.default_payment.expiry or "(empty)",
                observed.payment.expiry or "(empty)",
                passed,
            )
    else:
        # For user-specific payment, verify against user_info
        if task.cc_number:
            expected = re.sub(r"\D", "", task.cc_number)
            observed_val = re.sub(r"\D", "", observed.payment.card_number or "")
            passed = expected == observed_val
            add_check("payment_card_number_match", task.cc_number, observed.payment.card_number or "(empty)", passed)

        if task.cc_expires:
            expected = normalize_expiry(task.cc_expires)
            observed_val = normalize_expiry(observed.payment.expiry or "")
            passed = expected == observed_val
            add_check("payment_expiry_match", task.cc_expires, observed.payment.expiry or "(empty)", passed)

    return {"passed": len(failures) == 0, "failures": failures, "checks": checks}


def verify_delivery_info(task: BasicDeliveryInstructionParams, observed: ApplicationState) -> Dict[str, Any]:
    """Verify delivery instruction information matches expected values."""
    failures = []
    checks = []

    def add_check(name: str, expected: str, observed_val: str, passed: bool):
        checks.append({"check": name, "expected": expected, "observed": observed_val, "passed": passed})
        if not passed:
            failures.append(name)

    # Verify delivery recipient only when using user-specific address
    if task.delivery_address_type == "user" and task.expected_person_name:
        expected_recipient = normalize_string_for_comparison(task.expected_person_name)
        observed_recipient = normalize_string_for_comparison(observed.delivery.recipient_name)
        passed = expected_recipient == observed_recipient
        add_check("delivery_recipient_match", task.expected_person_name, observed.delivery.recipient_name, passed)
    else:
        # For default delivery, we don't verify specific recipient name
        add_check("delivery_address_type", "default (no specific recipient verification required)", "default", True)

    # Verify delivery date using proper date comparison
    # Prefer delivery_date_iso (from evaluation engine) if available, otherwise use delivery_date_text
    expected_date_iso = getattr(task, "delivery_date_iso", None)
    expected_date_text = task.delivery_date_text

    if expected_date_iso or expected_date_text:
        if observed.delivery.delivery_date:
            # Use ISO date if available (from evaluation engine), otherwise parse text
            expected_date_for_match = expected_date_iso if expected_date_iso else expected_date_text
            passed = dates_match(expected_date_for_match, observed.delivery.delivery_date)
            expected_date_str = expected_date_iso if expected_date_iso else parse_delivery_date_text(expected_date_text)
            add_check("delivery_date_match", expected_date_str, observed.delivery.delivery_date, passed)
        elif observed.delivery.delivery_date_text:
            # Fallback to text comparison if no parsed date available
            expected_text = normalize_string_for_comparison(expected_date_text or "")
            observed_text = normalize_string_for_comparison(observed.delivery.delivery_date_text)
            passed = expected_text == observed_text
            add_check(
                "delivery_date_text_match",
                expected_date_text or expected_date_iso,
                observed.delivery.delivery_date_text,
                passed,
            )
        else:
            # No delivery date info available
            add_check("delivery_date_missing", expected_date_iso or expected_date_text or "None", "None", False)

    # Verify property type
    if task.property_type:
        expected_property_type = normalize_string_for_comparison(task.property_type)
        observed_property_type = normalize_string_for_comparison(observed.delivery.property_type or "")

        if not observed_property_type:
            add_check("property_type_match", task.property_type, "None (not set)", False)
        else:
            passed = expected_property_type == observed_property_type
            add_check(
                "property_type_match",
                task.property_type,
                observed.delivery.property_type or "",
                passed,
            )

    # Verify delivery dropoff location
    if task.delivery_location:
        expected_location = normalize_string_for_comparison(task.delivery_location)
        observed_location = normalize_string_for_comparison(observed.delivery.dropoff_location or "")

        if not observed_location:
            add_check("delivery_location_match", task.delivery_location, "None (not set)", False)
        else:
            passed = expected_location == observed_location
            add_check(
                "delivery_location_match",
                task.delivery_location,
                observed.delivery.dropoff_location or "",
                passed,
            )

    return {"passed": len(failures) == 0, "failures": failures, "checks": checks}


def verify_confirmation_info(
    task: BasicDeliveryInstructionParams, observed: ApplicationState, agent_output: Optional[AgentOutput] = None
) -> Dict[str, Any]:
    """Verify confirmation information matches expected values."""
    failures = []
    checks = []

    def add_check(name: str, expected: str, observed_val: str, passed: bool):
        checks.append({"check": name, "expected": expected, "observed": observed_val, "passed": passed})
        if not passed:
            failures.append(name)

    # Check if confirmation page was reached
    passed = bool(observed.confirmation.reached)
    add_check("confirmation_reached", "true", str(observed.confirmation.reached), passed)

    # Check if confirmation text is present
    passed = bool(observed.confirmation.confirmation_text_present)
    add_check("confirmation_text_present", "true", str(observed.confirmation.confirmation_text_present), passed)

    # Check order number match if available
    if agent_output and agent_output.order_number:
        if observed.confirmation.order_number:
            expected = normalize_string_for_comparison(agent_output.order_number)
            observed_val = normalize_string_for_comparison(observed.confirmation.order_number)
            # Use substring matching - check if the actual order number is contained in the agent's response
            passed = observed_val in expected
            add_check("order_number_match", agent_output.order_number, observed.confirmation.order_number, passed)
            if not passed:
                failures.append("order_number_mismatch")
        else:
            # Agent provided order number but none found in app state
            add_check("order_number_match", agent_output.order_number, "None (not reached)", False)
            failures.append("order_number_not_found")
    elif agent_output and not agent_output.order_number:
        # Agent response provided but no order number given
        add_check("order_number_provided", "order number", "None provided", False)
        failures.append("order_number_not_provided")

    return {"passed": len(failures) == 0, "failures": failures, "checks": checks}


def verify_basic_checkout_complete(
    task: BasicDeliveryInstructionParams,
    observed: ApplicationState,
    agent_output: Optional[AgentOutput] = None,
    timed_out: bool = False,
) -> VerificationResult:
    """Main verification function for basic delivery instruction completion."""
    contact_result = verify_contact_info(task, observed)
    address_result = verify_address_info(task, observed)
    payment_result = verify_payment_info(task, observed)
    delivery_result = verify_delivery_info(task, observed)
    confirmation_result = verify_confirmation_info(task, observed, agent_output)

    # Collect all failures
    all_failures = []
    if not contact_result["passed"]:
        all_failures.extend(contact_result["failures"])
    if not address_result["passed"]:
        all_failures.extend(address_result["failures"])
    if not payment_result["passed"]:
        all_failures.extend(payment_result["failures"])
    if not delivery_result["passed"]:
        all_failures.extend(delivery_result["failures"])
    if not confirmation_result["passed"]:
        all_failures.extend(confirmation_result["failures"])
    if timed_out:
        all_failures.append("timeout")

    # Overall pass/fail
    passed = (
        contact_result["passed"]
        and address_result["passed"]
        and payment_result["passed"]
        and delivery_result["passed"]
        and confirmation_result["passed"]
        and not timed_out
    )

    details = {
        "contact": contact_result,
        "address": address_result,
        "payment": payment_result,
        "delivery_instruction": delivery_result,
        "confirmation": confirmation_result,
        "timeline": observed.timeline,
    }

    return VerificationResult(passed=passed, details=details, failures=all_failures)


def create_application_state_from_dict(data: Dict[str, Any]) -> ApplicationState:
    """Create ApplicationState from dictionary data."""
    shipping_info = data.get("shipping_info", {})
    payment_info = data.get("payment_info", {}) or {}
    delivery_instructions = data.get("delivery_instructions", {})
    default_shipping_info = data.get("default_shipping_info", {})
    default_payment_info = data.get("default_payment_info", {})
    confirmation_data = data.get("confirmation", {})
    timeline_data = data.get("timeline", [])

    # Extract contact info from shipping_info
    contact = ContactInfo(
        first_name=shipping_info.get("firstName"),
        last_name=shipping_info.get("lastName"),
        email=None,  # Not available in new format
        phone=shipping_info.get("phoneNumber"),
    )

    # Extract address info from shipping_info
    address = AddressInfo(
        street_address=shipping_info.get("streetAddress"),
        city=shipping_info.get("city"),
        state=shipping_info.get("state"),
        zip_code=shipping_info.get("zipCode"),
        country=None,  # Not available in new format
    )

    # Extract default address info from default_shipping_info
    default_address = None
    if default_shipping_info:
        default_address = AddressInfo(
            street_address=default_shipping_info.get("streetAddress"),
            city=default_shipping_info.get("city"),
            state=default_shipping_info.get("state"),
            zip_code=default_shipping_info.get("zipCode"),
            country=None,
        )

    # Extract payment info from payment_info
    payment = PaymentInfo(
        payment_method="user_card" if payment_info.get("cardNumber") else "default",
        card_type=None,  # Not available in new format
        card_number_masked=None,  # Not available in new format
        card_number=payment_info.get("cardNumber"),
        expiry=f"{payment_info.get('expiryMonth', '')}/{payment_info.get('expiryYear', '')}"
        if payment_info.get("expiryMonth") and payment_info.get("expiryYear")
        else None,
        used_default=payment_info.get("setAsDefault"),
    )

    # Extract default payment info from default_payment_info
    default_payment = None
    if default_payment_info:
        default_payment = PaymentInfo(
            payment_method="default",
            card_type=None,
            card_number_masked=None,
            card_number=default_payment_info.get("cardNumber"),
            expiry=f"{default_payment_info.get('expiryMonth', '')}/{default_payment_info.get('expiryYear', '')}"
            if default_payment_info.get("expiryMonth") and default_payment_info.get("expiryYear")
            else None,
            used_default=None,
        )

    # Extract delivery info from timeline and delivery_instructions
    # Look for delivery recipient and date in timeline
    delivery_recipient = None
    delivery_date_text = None
    delivery_date = None

    for event in timeline_data:
        if event.get("action") == "selected_delivery_recipient":
            delivery_recipient = event.get("data", {}).get("recipient")
        elif event.get("action") == "selected_delivery_date":
            delivery_date_text = event.get("data", {}).get("date_text")
            delivery_date = event.get("data", {}).get("date")

    delivery_instruction = DeliveryInstructionInfo(
        recipient_name=delivery_recipient,
        delivery_date_text=delivery_date_text,
        delivery_date=delivery_date,
        delivery_method=None,  # Not available in new format
        property_type=delivery_instructions.get("propertyType"),
        dropoff_location=delivery_instructions.get("dropoffLocation"),
        used_default_dropoff=None,  # Not available in new format
    )

    confirmation = ConfirmationInfo(
        reached=confirmation_data.get("reached", False),
        order_number=confirmation_data.get("order_number"),
        confirmation_text_present=confirmation_data.get("confirmation_text_present", False),
    )

    return ApplicationState(
        url=data.get("url"),
        contact=contact,
        address=address,
        payment=payment,
        delivery=delivery_instruction,
        confirmation=confirmation,
        timeline=timeline_data,
        default_address=default_address,
        default_payment=default_payment,
    )


def create_basic_checkout_params_from_task(task: Dict[str, Any]) -> BasicDeliveryInstructionParams:
    """Create BasicDeliveryInstructionParams from task dictionary."""
    values = task.get("values", {})
    user_info = values.get("user_info", {})

    # Extract delivery address type, payment method type, property type, and delivery location from values
    delivery_address_value = values.get("delivery_address", "default")
    payment_method_type = values.get("payment_method", "default")
    property_type = values.get("property_type", "House")
    delivery_location = values.get("delivery_location", "Front door")
    # Prefer delivery_date_iso from evaluation engine, fallback to delivery_date_text for backward compatibility
    delivery_date_iso = values.get("delivery_date_iso")
    delivery_date_text = values.get("delivery_date_text") or values.get("delivery_date")

    # Determine delivery address type
    # If delivery_address is "default", it's a default address
    # Otherwise, it's user-specific (person name)
    if delivery_address_value == "default":
        delivery_address_type = "default"
        expected_person_name = None
    else:
        delivery_address_type = "user"
        # The delivery_address value is the person's full name
        expected_person_name = delivery_address_value

    return BasicDeliveryInstructionParams(
        delivery_address_type=delivery_address_type,
        payment_method_type=payment_method_type,
        delivery_date_text=delivery_date_text,
        delivery_date_iso=delivery_date_iso,
        property_type=property_type,
        delivery_location=delivery_location,
        expected_person_name=expected_person_name,
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


# =========================
# Evaluator
# =========================


class InputDict(TypedDict, total=False):
    page: "Page"


class FinalResult(BaseModel):
    """Result model for Megamart basic delivery instruction task."""

    score: float
    reasoning: str
    current_state: dict[str, Any]
    ground_truth: Any


class HalluminateMegamartBasicDeliveryInstruction(BaseMetric):
    """Evaluator that runs verifier for Megamart basic delivery instruction task."""

    def __init__(self, task: str, values: dict[str, Any], user_metadata: UserMetadata) -> None:
        super().__init__()
        self.task = task
        self.values = values
        self.user_metadata = user_metadata
        # Create ground truth from task values (verifier expects {"task": task, "values": values})
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
            return FinalResult(
                score=0.0,
                reasoning="Failed (verification): " + "; ".join(result.failures),
                current_state=current_state,
                ground_truth=self._ground_truth,
            )

    @async_retry_with_exponential_backoff()
    async def _get_current_state(self) -> dict[str, Any]:
        return await self._page.evaluate("() => window.__MEGAMART_EVAL_STATE__")


def _evaluate_date_formula(formula_str: str, timezone: str, timestamp: int) -> datetime:
    """Safely evaluate a date formula string and return datetime object.

    Example:
        Input:  formula_str="now() + timedelta(days=2)", timezone="America/Los_Angeles", timestamp=1715702400
        Output: datetime object with timezone info
    """

    def now():
        return datetime.fromtimestamp(timestamp, tz=ZoneInfo(timezone))

    safe_globals = {"timedelta": timedelta, "now": now}
    result = eval(formula_str, safe_globals)
    if not isinstance(result, datetime):
        raise ValueError(f"Formula did not return a datetime: {formula_str}")
    return result


def _format_date_for_task(date_obj: datetime) -> str:
    """Format datetime to match Megamart frontend's date format.

    Example:
        Input:  datetime(2025, 11, 12, ...)
        Output: "Wednesday, Nov 12"
    """
    return date_obj.strftime("%A, %b %d")


def _render_values(values: dict[str, Any], timezone: str, timestamp: int) -> dict[str, Any]:
    """Render delivery_date_formula for basic_delivery_instruction tasks.

    Evaluates delivery_date_formula to get actual ISO date, while keeping delivery_date_text
    for verifier compatibility.

    IMPORTANT: The verifier's create_basic_checkout_params_from_task expects
    values.get("delivery_date", "tomorrow") to be the TEXT format (e.g., "2-days from now" or "2 days from now"),
    not the ISO date. So we:
    1. Evaluate formula to get ISO date → store in delivery_date_iso
    2. Keep delivery_date_text in values (verifier will read it via delivery_date field)
    3. For task rendering, we use delivery_date_iso
    """
    evaluated = dict(values)

    # If delivery_date_formula exists, evaluate it
    if "delivery_date_formula" in evaluated:
        formula = evaluated["delivery_date_formula"]
        if isinstance(formula, str) and ("now()" in formula or "timedelta" in formula):
            try:
                # Evaluate: "now() + timedelta(days=2)" → datetime object
                evaluated_datetime = _evaluate_date_formula(formula, timezone, timestamp)

                # Format the date to match frontend format (e.g., "Wednesday, Nov 12")
                formatted_date = _format_date_for_task(evaluated_datetime)

                # Store formatted date for task rendering
                evaluated["delivery_date"] = formatted_date

                # Store ISO date for reference
                evaluated["delivery_date_iso"] = evaluated_datetime.strftime("%Y-%m-%d")

                # The verifier expects delivery_date_description for text format (e.g., "2 days from now")
                # Keep delivery_date_description for verifier compatibility
                if "delivery_date_description" not in evaluated:
                    # If no description, use the formatted date as fallback
                    # But verifier expects text like "2 days from now", not "Wednesday, Nov 12"
                    pass

            except (ValueError, SyntaxError) as e:
                # If formula evaluation fails, keep original values
                print(f"Warning: Failed to evaluate delivery_date_formula '{formula}': {e}")
                # Fallback: use delivery_date_description if available
                if "delivery_date_description" in evaluated:
                    evaluated["delivery_date"] = evaluated["delivery_date_description"]

    return evaluated


def _render_task(task: str, values: dict[str, Any]) -> str:
    """Replace placeholders in task string.

    Placeholders: {delivery_date}, {delivery_address}, {property_type}, {delivery_location},
    {payment_method}, {user_info}.

    Example:
        Input task: "Make sure the delivery day is set to {delivery_date}..."
        Input values: {"delivery_date_iso": "2025-11-13", "property_type": "Business", ...}
        Output: "Make sure the delivery day is set to Thursday, November 13, 2025..."
    """
    result = task

    # Replace {delivery_date} with formatted date
    if "{delivery_date}" in result:
        # Use delivery_date (already formatted by _render_values) or fallback to description
        date_str = values.get("delivery_date") or values.get("delivery_date_description", "")
        if date_str:
            result = result.replace("{delivery_date}", date_str)

    # Replace {delivery_address}
    if "{delivery_address}" in result:
        delivery_address = values.get("delivery_address", "")
        if delivery_address == "default":
            result = result.replace("{delivery_address}", "the default shipping address")
        else:
            # For user-specific address, use the person's name (already in delivery_address)
            result = result.replace("{delivery_address}", delivery_address)

    # Replace {property_type}
    if "{property_type}" in result:
        property_type = values.get("property_type", "")
        result = result.replace("{property_type}", property_type)

    # Replace {delivery_location}
    if "{delivery_location}" in result:
        delivery_location = values.get("delivery_location", "")
        result = result.replace("{delivery_location}", delivery_location)

    # Replace {payment_method}
    if "{payment_method}" in result:
        payment_method = values.get("payment_method", "")
        if payment_method == "default":
            result = result.replace("{payment_method}", "the default payment method")
        else:
            result = result.replace("{payment_method}", payment_method)

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
    eval_target = get_import_path(HalluminateMegamartBasicDeliveryInstruction)
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
        "task_id": "halluminate_bench/megamart/basic_delivery_instruction/1",
        "task_generation_config_json": json.dumps(
            {
                "_target_": "halluminate_bench.megamart.basic_delivery_instruction.generate_task_config",
                "url": "https://megamart.halluminate.ai/?apiKey={HALLUMINATE_API_KEY}",
                "task": (
                    "Purchase this vase and deliver it to {delivery_address}. "
                    "For the delivery instructions, set the property type to {property_type} "
                    "and the dropoff location to {delivery_location}. "
                    "Make sure the delivery day is set to {delivery_date}. "
                    "For payment, use {payment_method}. "
                    "Your job is done when the order is successfully placed. "
                    "You may need to reference the following user info: {user_info}"
                ),
                "values": {
                    "delivery_address": "Carley Sanchez",
                    "property_type": "Business",
                    "delivery_location": "No preference",
                    "delivery_date_formula": "now() + timedelta(days=2)",
                    "delivery_date_text": "2-days from now",
                    "payment_method": "default",
                    "user_info": {
                        "Title": "Dr.",
                        "GivenName": "Carley",
                        "MiddleInitial": "J",
                        "Surname": "Sanchez",
                        "StreetAddress": "214 Layman Court",
                        "City": "Atlanta",
                        "State": "GA",
                        "StateFull": "Georgia",
                        "ZipCode": "30328",
                        "Country": "US",
                        "CountryFull": "United States",
                        "EmailAddress": "CarleyJSanchez@fleckens.hu",
                        "TelephoneNumber": "678-341-8484",
                        "TelephoneCountryCode": "1",
                        "CCType": "Visa",
                        "CCNumber": "4929597734332138",
                        "CVV2": "517",
                        "CCExpires": "5/2026",
                    },
                },
                "location": "San Francisco, CA, United States",
                "timezone": "America/Los_Angeles",
                "timestamp": None,
            }
        ),
        "env": "sim",
        "domain": "megamart",
        "l1_category": "e_commerce",
        "l2_category": "basic_delivery_instruction",
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
