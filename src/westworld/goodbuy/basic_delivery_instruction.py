import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Dict, List, Optional, TypedDict

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
    card_number_masked: Optional[str] = None
    card_number: Optional[str] = None
    expiry: Optional[str] = None
    expiry_date: Optional[str] = None
    cvv: Optional[str] = None
    used_default: Optional[bool] = None
    billing_address: Optional[Dict[str, str]] = None


@dataclass
class DeliveryInstructionInfo:
    """Delivery instruction information."""

    dropoff_location: Optional[str] = None
    property_type: Optional[str] = None
    delivery_notes: Optional[str] = None
    delivery_date: Optional[str] = None
    delivery_time: Optional[str] = None


@dataclass
class ConfirmationInfo:
    """Confirmation information for the delivery."""

    reached: bool = False
    order_number: Optional[str] = None
    confirmation_text_present: bool = False


@dataclass
class ApplicationState:
    """Current state of the delivery instruction application."""

    contact: ContactInfo
    address: AddressInfo
    payment: PaymentInfo
    delivery_instruction: DeliveryInstructionInfo
    confirmation: ConfirmationInfo
    timeline: List[Dict[str, Any]]
    cart: Optional[Any] = None
    url: Optional[str] = None
    delivery_date: Optional[str] = None


@dataclass
class BasicDeliveryInstructionParams:
    """Expected delivery instruction parameters from the basic delivery instruction task."""

    delivery_address: str  # "user" or specific address
    payment_method_type: str  # "default" or "user"
    delivery_date: str  # "today", "tomorrow", or specific date
    dropoff_location: str  # "front door", "back door", "garage", etc.
    property_type: str  # "house", "apartment", "condo", etc.
    delivery_notes: str  # "call on arrival", "leave at door", etc.
    task_description: str  # Original task description for conditional verification
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
    """Agent output information."""

    order_number: Optional[str] = None
    raw_response: Optional[str] = None
    other: Optional[Dict[str, Any]] = None


@dataclass
class VerificationResult:
    """Result of verification."""

    passed: bool
    details: Dict[str, Any]
    failures: List[str]


# =========================
# Verifier functions (copied from verifier)
# =========================


def normalize_string_for_comparison(text: str) -> str:
    """Normalize string for comparison by removing extra whitespace and converting to lowercase."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())


def get_last4_digits(card_number: Optional[str]) -> str:
    """Extract last 4 digits from card number."""
    if not card_number:
        return ""
    digits = re.sub(r"\D", "", card_number)
    return digits[-4:] if len(digits) >= 4 else ""


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


def parse_delivery_date(date_text: str) -> str:
    """Parse delivery date text into a standardized format."""
    if not date_text:
        return ""

    date_text = date_text.lower().strip()

    # Handle relative dates
    if "today" in date_text:
        return datetime.now().strftime("%Y-%m-%d")
    elif "tomorrow" in date_text:
        return (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    elif "yesterday" in date_text:
        return (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    # Handle specific dates (MM/DD/YYYY, MM-DD-YYYY, etc.)
    date_patterns = [
        r"(\d{1,2})/(\d{1,2})/(\d{4})",  # MM/DD/YYYY
        r"(\d{1,2})-(\d{1,2})-(\d{4})",  # MM-DD-YYYY
        r"(\d{4})-(\d{1,2})-(\d{1,2})",  # YYYY-MM-DD
    ]

    for pattern in date_patterns:
        match = re.search(pattern, date_text)
        if match:
            if len(match.group(1)) == 4:  # YYYY-MM-DD format
                year, month, day = match.groups()
            else:  # MM/DD/YYYY or MM-DD-YYYY format
                month, day, year = match.groups()
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"

    return date_text


def verify_delivery_instruction_selected(observed: ApplicationState) -> Dict[str, Any]:
    """Verify that delivery instruction was selected."""
    failures = []
    checks = []

    def add_check(name: str, expected: str, observed_val: str, passed: bool):
        checks.append({"check": name, "expected": expected, "observed": observed_val, "passed": passed})
        if not passed:
            failures.append(name)

    # Check if delivery instruction was selected by looking at timeline
    delivery_selected = False
    for event in observed.timeline:
        if (
            event.get("action") == "changed_delivery_method"
            and event.get("data", {}).get("delivery_method") == "shipping"
        ):
            delivery_selected = True
            break
        # Also check for other delivery-related events
        elif event.get("action") in ["set_delivery_method", "select_delivery", "delivery_selected"]:
            delivery_selected = True
            break

    # If no timeline events found, check if we're on a delivery-related page or have delivery data
    if not delivery_selected:
        # Check if we have delivery data in the state
        if observed.delivery_instruction and (
            observed.delivery_instruction.dropoff_location or observed.delivery_instruction.delivery_notes
        ):
            delivery_selected = True
        # Check if URL indicates delivery
        elif observed.url and "delivery" in observed.url.lower():
            delivery_selected = True
        # Check if there are any cart items with shipping delivery method
        elif observed.cart and isinstance(observed.cart, dict) and observed.cart.get("items"):
            for item in observed.cart["items"]:
                if isinstance(item, dict) and item.get("fulfillment", {}).get("mode") == "ship":
                    delivery_selected = True
                    break

    add_check(
        "delivery_method_selected",
        "delivery method selected",
        "delivery selected" if delivery_selected else "delivery not selected",
        delivery_selected,
    )

    return {"passed": len(failures) == 0, "failures": failures, "checks": checks}


def verify_delivery_instruction_details(
    task: BasicDeliveryInstructionParams, observed: ApplicationState
) -> Dict[str, Any]:
    """Verify delivery instruction details match expected values."""
    failures = []
    checks = []

    def add_check(name: str, expected: str, observed_val: str, passed: bool):
        checks.append({"check": name, "expected": expected, "observed": observed_val, "passed": passed})
        if not passed:
            failures.append(name)

    # Verify dropoff location
    if task.dropoff_location:
        expected_dropoff = normalize_string_for_comparison(task.dropoff_location)
        observed_dropoff = normalize_string_for_comparison(observed.delivery_instruction.dropoff_location or "")

        # If not in delivery_instruction object, try to get from timeline events
        if not observed_dropoff:
            for event in observed.timeline:
                if event.get("action") == "set_delivery_instruction":
                    event_data = event.get("data", {})
                    if isinstance(event_data, dict):
                        observed_dropoff = normalize_string_for_comparison(
                            event_data.get("dropoff_location") or event_data.get("dropoffLocation") or ""
                        )
                        break

        passed = expected_dropoff == observed_dropoff
        add_check("dropoff_location", expected_dropoff, observed_dropoff, passed)

    # Verify property type
    if task.property_type:
        expected_property = normalize_string_for_comparison(task.property_type)
        observed_property = normalize_string_for_comparison(observed.delivery_instruction.property_type or "")

        # If not in delivery_instruction object, try to get from timeline events
        if not observed_property:
            for event in observed.timeline:
                if event.get("action") == "set_delivery_instruction":
                    event_data = event.get("data", {})
                    if isinstance(event_data, dict):
                        observed_property = normalize_string_for_comparison(
                            event_data.get("property_type") or event_data.get("propertyType") or ""
                        )
                        break

        passed = expected_property == observed_property
        add_check("property_type", expected_property, observed_property, passed)

    # Verify delivery notes
    if task.delivery_notes:
        expected_notes = normalize_string_for_comparison(task.delivery_notes)
        observed_notes = normalize_string_for_comparison(observed.delivery_instruction.delivery_notes or "")

        # If not in delivery_instruction object, try to get from timeline events
        if not observed_notes:
            for event in observed.timeline:
                if event.get("action") == "set_delivery_instruction":
                    event_data = event.get("data", {})
                    if isinstance(event_data, dict):
                        observed_notes = normalize_string_for_comparison(
                            event_data.get("delivery_notes") or event_data.get("deliveryNotes") or ""
                        )
                        break

        passed = expected_notes == observed_notes
        add_check("delivery_notes", expected_notes, observed_notes, passed)

    return {"passed": len(failures) == 0, "failures": failures, "checks": checks}


def verify_payment_details(task: BasicDeliveryInstructionParams, observed: ApplicationState) -> Dict[str, Any]:
    """Verify payment details match task requirements."""
    failures = []
    checks = []

    def add_check(name: str, expected: str, observed_val: str, passed: bool):
        checks.append({"check": name, "expected": expected, "observed": observed_val, "passed": passed})
        if not passed:
            failures.append(name)

    # Only verify specific card details if payment method is "user"
    if task.payment_method_type == "user":
        # Verify CCType from task against observed
        if task.cc_type:
            expected_cc_type = task.cc_type
            observed_cc_type = observed.payment.card_type or ""
            passed = expected_cc_type.lower() == observed_cc_type.lower()
            add_check("cc_type", f"CCType: {expected_cc_type}", observed_cc_type, passed)

        # Verify CCNumber from task against observed (full number)
        if task.cc_number:
            expected_cc_number = task.cc_number
            # Try full card number first, then masked
            if observed.payment.card_number:
                observed_cc_number = observed.payment.card_number
                passed = expected_cc_number == observed_cc_number
            elif observed.payment.card_number_masked:
                # If only masked is available, verify last 4 digits match
                expected_last4 = get_last4_digits(expected_cc_number)
                observed_last4 = get_last4_digits(observed.payment.card_number_masked)
                observed_cc_number = observed.payment.card_number_masked
                passed = expected_last4 == observed_last4
            else:
                observed_cc_number = ""
                passed = False
            add_check("cc_number", f"CCNumber: {expected_cc_number}", observed_cc_number, passed)

        # Verify CVV2 from task against observed
        if task.cvv2:
            expected_cvv2 = task.cvv2
            observed_cvv2 = observed.payment.cvv or ""
            passed = expected_cvv2 == observed_cvv2
            add_check("cvv2", f"CVV2: {expected_cvv2}", observed_cvv2, passed)

        # Verify CCExpires from task against observed (with normalization)
        if task.cc_expires:
            expected_expires = normalize_expiry(task.cc_expires)
            observed_expires = normalize_expiry(observed.payment.expiry_date or observed.payment.expiry or "")
            passed = expected_expires == observed_expires
            add_check("cc_expires", f"CCExpires: {expected_expires}", observed_expires, passed)

    # Check payment method type
    if task.payment_method_type == "default":
        # For default payment method, verify user actually used the default
        if observed.payment.used_default is not None:
            passed = observed.payment.used_default
            add_check("used_default_payment", "default payment method used", str(observed.payment.used_default), passed)
        else:
            # Legacy support: fallback to checking payment method type
            if observed.payment.payment_method:
                observed_method = observed.payment.payment_method.lower()
                passed = "default" in observed_method or "saved" in observed_method
                add_check("payment_method_type", "default payment method", observed.payment.payment_method, passed)
            else:
                add_check("payment_method_type", "default payment method", "default", True)
    else:
        # For user-specified payment, verify the specific card details match user's card
        if task.cc_number:
            # Try to use full card number if available, otherwise use masked
            if observed.payment.card_number:
                # Verify full card number matches
                passed = task.cc_number == observed.payment.card_number
                add_check("card_number", f"user's card: {task.cc_number}", observed.payment.card_number, passed)
            elif observed.payment.card_number_masked:
                # If only masked is available, verify last 4 digits match
                expected_last4 = get_last4_digits(task.cc_number)
                observed_last4 = get_last4_digits(observed.payment.card_number_masked)
                passed = expected_last4 == observed_last4
                add_check(
                    "card_number",
                    f"user's card ending in {expected_last4}",
                    f"ending in {observed_last4}",
                    passed,
                )

            # Verify card type if available
            if task.cc_type and observed.payment.card_type:
                expected_type = task.cc_type
                observed_type = observed.payment.card_type
                passed = expected_type.lower() == observed_type.lower()
                add_check("card_type", f"user's card type: {task.cc_type}", observed.payment.card_type, passed)

            # Verify expiry if available
            if task.cc_expires and (observed.payment.expiry_date or observed.payment.expiry):
                expected_expiry = normalize_expiry(task.cc_expires)
                observed_expiry = normalize_expiry(observed.payment.expiry_date or observed.payment.expiry or "")
                passed = expected_expiry == observed_expiry
                add_check("card_expiry", f"user's card expiry: {task.cc_expires}", observed_expiry, passed)

    return {"passed": len(failures) == 0, "failures": failures, "checks": checks}


def verify_billing_address(task: BasicDeliveryInstructionParams, observed: ApplicationState) -> Dict[str, Any]:
    """Verify billing address matches user's address if required by task."""
    failures = []
    checks = []

    def add_check(name: str, expected: str, observed_val: str, passed: bool):
        checks.append({"check": name, "expected": expected, "observed": observed_val, "passed": passed})
        if not passed:
            failures.append(name)

    # Check if billing address verification is required
    # For user payment method, billing address should match user's address
    requires_billing_address = task.payment_method_type == "user"

    if not requires_billing_address:
        # Task doesn't require billing address verification
        add_check("billing_address_required", "billing address verification not required", "not required", True)
        return {"passed": True, "failures": [], "checks": checks}

    # Get user's address from task
    user_address = {"street": task.street_address, "city": task.city, "state": task.state, "zip": task.zip_code}

    # Get observed billing address from payment info, or use delivery address if billing address not available
    observed_billing = (
        observed.payment.billing_address
        if hasattr(observed.payment, "billing_address") and observed.payment.billing_address
        else {}
    )

    # If billing address is not available, use delivery address (billing should match delivery for user payment)
    if not observed_billing and observed.address:
        observed_billing = {
            "street": observed.address.street_address,
            "city": observed.address.city,
            "state": observed.address.state,
            "zip": observed.address.zip_code,
        }

    # Verify street address
    if user_address["street"]:
        expected_street = normalize_string_for_comparison(user_address["street"])
        observed_street = normalize_string_for_comparison(observed_billing.get("street", ""))
        passed = expected_street == observed_street
        add_check(
            "billing_street", f"billing street: {user_address['street']}", observed_billing.get("street", ""), passed
        )

    # Verify city
    if user_address["city"]:
        expected_city = normalize_string_for_comparison(user_address["city"])
        observed_city = normalize_string_for_comparison(observed_billing.get("city", ""))
        passed = expected_city == observed_city
        add_check("billing_city", f"billing city: {user_address['city']}", observed_billing.get("city", ""), passed)

    # Verify state
    if user_address["state"]:
        expected_state = normalize_string_for_comparison(user_address["state"])
        observed_state = normalize_string_for_comparison(observed_billing.get("state", ""))
        passed = expected_state == observed_state
        add_check("billing_state", f"billing state: {user_address['state']}", observed_billing.get("state", ""), passed)

    # Verify zip code
    if user_address["zip"]:
        expected_zip = normalize_string_for_comparison(user_address["zip"])
        observed_zip = normalize_string_for_comparison(observed_billing.get("zip", ""))
        passed = expected_zip == observed_zip
        add_check("billing_zip", f"billing zip: {user_address['zip']}", observed_billing.get("zip", ""), passed)

    return {"passed": len(failures) == 0, "failures": failures, "checks": checks}


def verify_order_completion(observed: ApplicationState, agent_output: Optional[AgentOutput] = None) -> Dict[str, Any]:
    """Verify order was successfully completed."""
    failures = []
    checks = []

    def add_check(name: str, expected: str, observed_val: str, passed: bool):
        checks.append({"check": name, "expected": expected, "observed": observed_val, "passed": passed})
        if not passed:
            failures.append(name)

    # Check if confirmation page was reached
    passed = bool(observed.confirmation.reached)
    add_check("confirmation_reached", "order confirmation page reached", str(observed.confirmation.reached), passed)

    # Check if confirmation text is present
    passed = bool(observed.confirmation.confirmation_text_present)
    add_check(
        "confirmation_text_present",
        "order confirmation text present",
        str(observed.confirmation.confirmation_text_present),
        passed,
    )

    # Check order number match if available
    if agent_output and agent_output.order_number:
        if observed.confirmation.order_number:
            expected = normalize_string_for_comparison(agent_output.order_number)
            observed_val = normalize_string_for_comparison(observed.confirmation.order_number)
            # Use substring matching - check if the actual order number is contained in the agent's response
            passed = observed_val in expected
            add_check(
                "order_number_match",
                f"order number: {agent_output.order_number}",
                observed.confirmation.order_number,
                passed,
            )
            if not passed:
                failures.append("order_number_mismatch")
        else:
            # Agent provided order number but none found in app state
            add_check("order_number_match", f"order number: {agent_output.order_number}", "None (not reached)", False)
            failures.append("order_number_not_found")
    elif agent_output and not agent_output.order_number:
        # Agent response provided but no order number given
        add_check("order_number_provided", "order number provided by agent", "None provided", False)
        failures.append("order_number_not_provided")

    return {"passed": len(failures) == 0, "failures": failures, "checks": checks}


def verify_basic_delivery_instruction_complete(
    task: BasicDeliveryInstructionParams,
    observed: ApplicationState,
    agent_output: Optional[AgentOutput] = None,
    timed_out: bool = False,
) -> VerificationResult:
    """Main verification function for basic delivery instruction completion."""
    delivery_method_result = verify_delivery_instruction_selected(observed)
    delivery_instruction_result = verify_delivery_instruction_details(task, observed)
    payment_details_result = verify_payment_details(task, observed)
    billing_address_result = verify_billing_address(task, observed)
    order_completion_result = verify_order_completion(observed, agent_output)

    # Collect all failures
    all_failures = []
    if not delivery_method_result["passed"]:
        all_failures.extend(delivery_method_result["failures"])
    if not delivery_instruction_result["passed"]:
        all_failures.extend(delivery_instruction_result["failures"])
    if not payment_details_result["passed"]:
        all_failures.extend(payment_details_result["failures"])
    if not billing_address_result["passed"]:
        all_failures.extend(billing_address_result["failures"])
    if not order_completion_result["passed"]:
        all_failures.extend(order_completion_result["failures"])
    if timed_out:
        all_failures.append("timeout")

    # Overall pass/fail
    passed = (
        delivery_method_result["passed"]
        and delivery_instruction_result["passed"]
        and payment_details_result["passed"]
        and billing_address_result["passed"]
        and order_completion_result["passed"]
        and not timed_out
    )

    details = {
        "delivery_method": delivery_method_result,
        "delivery_instruction": delivery_instruction_result,
        "payment_details": payment_details_result,
        "billing_address": billing_address_result,
        "order_completion": order_completion_result,
        "timeline": observed.timeline,
    }

    return VerificationResult(passed=passed, details=details, failures=all_failures)


def create_application_state_from_dict(data: Dict[str, Any]) -> ApplicationState:
    """Create ApplicationState from dictionary data."""
    shipping_info = data.get("shipping_info", {})
    payment_info = data.get("payment_info", {}) or {}
    delivery_instructions = data.get("delivery_instructions", {})
    confirmation_data = data.get("confirmation", {})
    timeline_data = data.get("timeline", [])
    cart_data = data.get("cart", {})

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

    # Extract payment info from payment_info
    masked_card = payment_info.get("maskedCard")
    payment = PaymentInfo(
        payment_method=payment_info.get("method", "default"),
        card_type=payment_info.get("cardType"),  # Now available in frontend
        card_number_masked=masked_card,  # Use masked card from frontend
        card_number=payment_info.get("cardNumber"),  # Now available in frontend
        expiry=f"{payment_info.get('expiryMonth', '')}/{payment_info.get('expiryYear', '')}"
        if payment_info.get("expiryMonth") and payment_info.get("expiryYear")
        else None,
        expiry_date=payment_info.get("expiryDate"),  # Now available in frontend
        cvv=payment_info.get("cvv"),  # Now available in frontend
        used_default=payment_info.get("setAsDefault"),
        billing_address=None,
    )

    # Extract delivery instruction info
    delivery_instruction = DeliveryInstructionInfo(
        dropoff_location=delivery_instructions.get("dropoff_location"),
        property_type=delivery_instructions.get("property_type"),
        delivery_notes=delivery_instructions.get("delivery_notes") or delivery_instructions.get("deliveryNotes"),
        delivery_date=delivery_instructions.get("delivery_date") or delivery_instructions.get("deliveryDate"),
        delivery_time=delivery_instructions.get("delivery_time") or delivery_instructions.get("deliveryTime"),
    )

    # Handle confirmation data - can be bool or dict
    if isinstance(confirmation_data, bool):
        confirmation = ConfirmationInfo(reached=confirmation_data)
    elif isinstance(confirmation_data, dict):
        confirmation = ConfirmationInfo(
            reached=confirmation_data.get("reached", False),
            order_number=confirmation_data.get("order_number"),
            confirmation_text_present=confirmation_data.get("confirmation_text_present", False),
        )
    else:
        confirmation = ConfirmationInfo()

    # Extract cart data
    cart_items = []
    if cart_data and cart_data.get("items"):
        for item_data in cart_data["items"]:
            cart_item = {
                "id": item_data.get("id"),
                "title": item_data.get("title"),
                "fulfillment": item_data.get("fulfillment", {}),
            }
            cart_items.append(cart_item)

    cart_info = {"items": cart_items} if cart_items else None

    return ApplicationState(
        url=data.get("url"),
        contact=contact,
        address=address,
        payment=payment,
        delivery_instruction=delivery_instruction,
        confirmation=confirmation,
        timeline=timeline_data,
        cart=cart_info,
        delivery_date=data.get("delivery_date"),
    )


def create_basic_delivery_instruction_params_from_task(task: Dict[str, Any]) -> BasicDeliveryInstructionParams:
    """Create BasicDeliveryInstructionParams from task dictionary."""
    values = task.get("values", {})
    user_info = values.get("user_info", {})

    # Extract parameters from task description
    task_description = task.get("task", "")

    # Parse delivery address
    delivery_address = "user"  # default
    if "deliver it to" in task_description.lower():
        # Extract delivery address if specified
        match = re.search(r"deliver it to ([^.]+?)(?:\.|$)", task_description)
        if match:
            delivery_address = match.group(1).strip()

    # Parse dropoff location
    dropoff_location = "front door"  # default
    if "dropoff location" in task_description.lower():
        # Try pattern: "dropoff location to garage and property type to apartment"
        match = re.search(r"dropoff location to ([^.]+?)(?: and property type|\.|$)", task_description)
        if match:
            full_text = match.group(1).strip()

            # Check if this contains " and " (like "house and front door")
            if " and " in full_text:
                parts = full_text.split(" and ")
                if len(parts) == 2:
                    # First part is property type, second part is dropoff location
                    property_type = parts[0].strip()
                    dropoff_location = parts[1].strip()
                else:
                    dropoff_location = full_text
            else:
                dropoff_location = full_text

    # Parse property type (only if not already set from dropoff location parsing)
    if "property_type" not in locals():
        property_type = "house"  # default
        if "property type" in task_description.lower():
            # Try pattern: "property type to apartment"
            match = re.search(r"property type to ([^.]+?)(?:\.|$)", task_description)
            if match:
                property_type = match.group(1).strip()

    # Parse delivery notes
    delivery_notes = ""  # default
    if "delivery notes" in task_description.lower():
        # Try pattern: "delivery notes to leave at door"
        match = re.search(r"delivery notes to ([^.]+?)(?:\.|$)", task_description)
        if match:
            delivery_notes = match.group(1).strip()

    # Parse payment method type
    payment_method_type = "default"  # default for most delivery tasks
    if "use the default payment method" in task_description.lower():
        payment_method_type = "default"
    elif "card ending in" in task_description.lower() or "use user's card" in task_description.lower():
        payment_method_type = "user"

    params = BasicDeliveryInstructionParams(
        delivery_address=delivery_address,
        payment_method_type=payment_method_type,
        delivery_date="",  # Not used anymore
        dropoff_location=dropoff_location,
        property_type=property_type,
        delivery_notes=delivery_notes,
        task_description=task.get("task", ""),
        # User info
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

    return params


class InputDict(TypedDict, total=False):
    page: "Page"


class FinalResult(BaseModel):
    """Result model for basic delivery instruction task."""

    score: float
    reasoning: str
    current_state: dict[str, Any]
    ground_truth: BasicDeliveryInstructionParams


class HalluminateGoodBuyBasicDeliveryInstruction(BaseMetric):
    """Evaluator that runs verifier for delivery instruction task."""

    def __init__(self, task: str, values: dict[str, Any], user_metadata: UserMetadata) -> None:
        super().__init__()
        self.task = task
        self.values = values
        self.user_metadata = user_metadata
        # Create ground truth from task values (verifier expects {"task": task, "values": values})
        self._ground_truth = create_basic_delivery_instruction_params_from_task({"task": task, "values": values})
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
        result = verify_basic_delivery_instruction_complete(self._ground_truth, state_obj)

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
            for section_name, section_result in result.details.items():
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
        return await self._page.evaluate("() => window.__BESTBUY_EVAL_STATE__")


def _render_values(values: dict[str, Any], timezone: str, timestamp: int) -> dict[str, Any]:
    """Render values for basic_delivery_instruction tasks.

    Delivery instruction tasks don't use delivery_date anymore.
    This is a no-op for now, but kept for consistency with other evaluators.
    """
    return dict(values)


def _render_task(task: str, values: dict[str, Any]) -> str:
    """Replace {user_info} placeholder in task text."""
    result = task

    # Replace {user_info} with string representation
    if "{user_info}" in result:
        user_info = values.get("user_info", {})
        result = result.replace("{user_info}", str(user_info))

    return result


def generate_task_config(
    url: str, task: str, values: dict[str, Any], location: str, timezone: str, timestamp: int | None = None
) -> BaseTaskConfig:
    """Generate a task config from raw parameters."""
    # Replace the API key in the URL
    url = url.replace("{HALLUMINATE_API_KEY}", os.getenv("HALLUMINATE_API_KEY"))

    # Get the task start timestamp
    timestamp = int(datetime.now().timestamp()) if timestamp is None else timestamp

    # Create the user metadata
    user_metadata = UserMetadata(timestamp=timestamp, location=location, timezone=timezone)

    # Render the values
    values = _render_values(values, timezone, timestamp)

    # Render the task (replace placeholders with formatted values)
    task = _render_task(task, values)

    # Create the eval config
    eval_target = get_import_path(HalluminateGoodBuyBasicDeliveryInstruction)
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
        "task_id": "halluminate_bench/goodbuy/basic_delivery_instruction/1",
        "task_generation_config_json": json.dumps(
            {
                "_target_": "halluminate_bench.goodbuy.basic_delivery_instruction.generate_task_config",
                "url": "https://good-buy.halluminate.ai/?apiKey={HALLUMINATE_API_KEY}",
                "task": (
                    "Purchase the items in my cart and deliver them to Hannah Lilly. "
                    "Set the delivery instruction dropoff location to front door and property type to apartment. "
                    "Set the delivery notes to call on arrival. For payment, use Hannah Lilly's payment info. "
                    "Your job is done when the order is successfully placed. Use guest checkout. "
                    "User info: {user_info}."
                ),
                "values": {
                    "delivery_address": "user",
                    "dropoff_location": "front door",
                    "property_type": "apartment",
                    "delivery_notes": "call on arrival",
                    "payment_method": "user",
                    "user_info": {
                        "Title": "Mrs.",
                        "GivenName": "Hannah",
                        "MiddleInitial": "C",
                        "Surname": "Lilly",
                        "StreetAddress": "322 Davis Place",
                        "City": "Bloomville",
                        "State": "OH",
                        "StateFull": "Ohio",
                        "ZipCode": "44818",
                        "Country": "US",
                        "CountryFull": "United States",
                        "EmailAddress": "HannahCLilly@rhyta.com",
                        "TelephoneNumber": "419-284-4231",
                        "TelephoneCountryCode": "1",
                        "Birthday": "8/30/1983",
                        "Age": "41",
                        "CCType": "MasterCard",
                        "CCNumber": "5370236131284763",
                        "CVV2": "433",
                        "CCExpires": "1/2028",
                    },
                },
                "location": "San Francisco, CA, United States",
                "timezone": "America/Los_Angeles",
                "timestamp": None,
            }
        ),
        "env": "sim",
        "domain": "goodbuy",
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
