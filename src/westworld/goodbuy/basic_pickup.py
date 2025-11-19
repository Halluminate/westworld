import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
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
    """Contact information for the pickup."""

    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None


@dataclass
class AddressInfo:
    """Address information for the pickup."""

    street_address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    country: Optional[str] = None


@dataclass
class PaymentInfo:
    """Payment information for the pickup."""

    payment_method: Optional[str] = None
    card_type: Optional[str] = None
    card_number_masked: Optional[str] = None
    card_number: Optional[str] = None  # Full card number for verification
    expiry: Optional[str] = None
    expiry_date: Optional[str] = None  # Alternative field name
    cvv: Optional[str] = None
    used_default: Optional[bool] = None
    billing_address: Optional[Dict[str, str]] = None


@dataclass
class AlternatePickupPerson:
    """Alternate pickup person information."""

    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None


@dataclass
class PickupInfo:
    """Pickup information for the order."""

    store_name: Optional[str] = None
    store_address: Optional[str] = None
    pickup_person_first_name: Optional[str] = None
    pickup_person_last_name: Optional[str] = None
    pickup_person_email: Optional[str] = None
    pickup_person_phone: Optional[str] = None
    alternate_pickup_person: Optional[AlternatePickupPerson] = None
    pickup_date: Optional[str] = None  # actual date in YYYY-MM-DD format
    pickup_time: Optional[str] = None  # "after 11am", "between 2pm-4pm", etc.
    pickup_method: Optional[str] = None  # "curbside" or "in-store"


@dataclass
class ConfirmationInfo:
    """Confirmation information for the pickup."""

    reached: bool = False
    order_number: Optional[str] = None
    confirmation_text_present: bool = False


@dataclass
class CartItem:
    """Cart item information."""

    id: Optional[str] = None
    title: Optional[str] = None
    fulfillment: Optional[Dict[str, Any]] = None


@dataclass
class CartInfo:
    """Cart information."""

    items: List[CartItem] = None


@dataclass
class ApplicationState:
    """Current state of the pickup application."""

    contact: ContactInfo
    address: AddressInfo
    payment: PaymentInfo
    pickup: PickupInfo
    confirmation: ConfirmationInfo
    timeline: List[Dict[str, Any]]
    cart: Optional[CartInfo] = None
    url: Optional[str] = None


@dataclass
class BasicPickupParams:
    """Expected pickup parameters from the basic pickup task."""

    payment_method_type: str  # "default" or "user"
    pickup_person_name: str  # "John Doe" - the primary pickup person
    pickup_method: str  # "curbside" or "in-store"
    task_description: str  # Original task description for conditional verification
    expected_store_name: Optional[str] = None  # Expected store name
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
# Verifier functions (copied from verifier)
# =========================


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


def verify_pickup_method_selected(observed: ApplicationState, expected_method: Optional[str] = None) -> Dict[str, Any]:
    """Verify that pickup method was selected and matches expected method."""
    failures = []
    checks = []

    def add_check(name: str, expected: str, observed_val: str, passed: bool):
        checks.append({"check": name, "expected": expected, "observed": observed_val, "passed": passed})
        if not passed:
            failures.append(name)

    # Check if pickup method was selected by looking at timeline
    pickup_selected = False
    selected_method = None

    for event in observed.timeline:
        if (
            event.get("action") == "changed_delivery_method"
            and event.get("data", {}).get("delivery_method") == "pickup"
        ):
            pickup_selected = True
            # Don't break here - continue looking for specific pickup method
        # Also check for other pickup-related events
        elif event.get("action") in ["set_pickup_method", "select_pickup", "pickup_selected"]:
            pickup_selected = True
            selected_method = event.get("data", {}).get("pickup_method")
            break
        elif event.get("action") in ["select_curbside", "curbside_selected"]:
            pickup_selected = True
            selected_method = "curbside"
            break
        elif event.get("action") in ["select_in_store", "in_store_selected"]:
            pickup_selected = True
            selected_method = "in-store"
            break

    # If no timeline events found, check if we're on a pickup-related page or have pickup data
    if not pickup_selected:
        # Check if we have pickup data in the state
        if observed.pickup and (observed.pickup.pickup_method or observed.pickup.store_name):
            pickup_selected = True
            # If pickup_method is None, blank, or empty string from global state,
            # treat it as in-store (default frontend behavior)
            pickup_method_from_state = observed.pickup.pickup_method
            if pickup_method_from_state and pickup_method_from_state.strip():
                selected_method = pickup_method_from_state
            # If blank/None from global state, will default to in-store below
        # Check if URL indicates pickup
        elif observed.url and "pickup" in observed.url.lower():
            pickup_selected = True
        # Check if there are any cart items with pickup delivery method
        elif observed.cart and observed.cart.items:
            for item in observed.cart.items:
                if item.fulfillment and item.fulfillment.get("mode") == "pickup":
                    pickup_selected = True
                    break

    # Quick fix: If pickup is selected but no method found in global state (None/blank),
    # default to "in-store" since that's the default in the frontend.
    # However, if curbside was explicitly selected, it should be in timeline events.
    method_is_blank = selected_method is None or (isinstance(selected_method, str) and not selected_method.strip())
    if pickup_selected and method_is_blank:
        # Double-check timeline for any curbside selection we might have missed
        found_curbside = False
        for event in observed.timeline:
            if event.get("action") in ["select_curbside", "curbside_selected", "set_pickup_method"]:
                event_data = event.get("data", {})
                if isinstance(event_data, dict):
                    method = event_data.get("pickup_method")
                    if method and "curbside" in method.lower():
                        found_curbside = True
                        selected_method = "curbside"
                        break
        # If no curbside found, default to in-store (frontend default)
        if not found_curbside:
            selected_method = "in-store"

    add_check(
        "pickup_method_selected",
        "pickup method selected",
        "pickup selected" if pickup_selected else "pickup not selected",
        pickup_selected,
    )

    # If we have an expected method, verify it matches
    if expected_method and pickup_selected:
        method_matches = selected_method == expected_method
        add_check(
            "pickup_method_match",
            f"pickup method: {expected_method}",
            f"pickup method: {selected_method or 'in-store'}",
            method_matches,
        )

    return {"passed": len(failures) == 0, "failures": failures, "checks": checks}


def verify_pickup_person(task: BasicPickupParams, observed: ApplicationState) -> Dict[str, Any]:
    """Verify pickup person information matches expected values."""
    failures = []
    checks = []

    def add_check(name: str, expected: str, observed_val: str, passed: bool):
        checks.append({"check": name, "expected": expected, "observed": observed_val, "passed": passed})
        if not passed:
            failures.append(name)

    # Verify pickup person name matches task requirement
    if task.pickup_person_name:
        name_parts = task.pickup_person_name.split()
        expected_first = name_parts[0]
        expected_last = " ".join(name_parts[1:]) if len(name_parts) > 1 else ""

        observed_first = observed.pickup.pickup_person_first_name or ""
        observed_last = observed.pickup.pickup_person_last_name or ""

        # If we don't have pickup person data in the main pickup object, try to get it from timeline
        if not observed_first and not observed_last:
            for event in observed.timeline:
                if event.get("action") == "set_pickup_person":
                    event_data = event.get("data", {})
                    if isinstance(event_data, dict):
                        observed_first = event_data.get("first_name") or event_data.get("firstName") or ""
                        observed_last = event_data.get("last_name") or event_data.get("lastName") or ""
                        break

        first_name_passed = expected_first == observed_first
        add_check(
            "pickup_person_first_name",
            f"pickup person first name: {name_parts[0]}",
            observed.pickup.pickup_person_first_name or "",
            first_name_passed,
        )

        if expected_last:
            last_name_passed = expected_last == observed_last
            add_check(
                "pickup_person_last_name",
                f"pickup person last name: {' '.join(name_parts[1:])}",
                observed.pickup.pickup_person_last_name or "",
                last_name_passed,
            )

    # Verify pickup person email matches expected value
    expected_email = task.email  # From task file
    observed_email = observed.pickup.pickup_person_email or ""

    # If not in pickup object, try to get from timeline events
    if not observed_email:
        for event in observed.timeline:
            if event.get("action") == "set_pickup_person":
                event_data = event.get("data", {})
                if isinstance(event_data, dict):
                    observed_email = event_data.get("email") or event_data.get("emailAddress") or ""
                    break

    email_passed = expected_email == observed_email if expected_email else bool(observed_email)
    add_check("pickup_person_email", f"pickup person email: {expected_email}", observed_email, email_passed)

    # Verify pickup person phone matches expected value
    expected_phone = task.phone  # From task file
    observed_phone = observed.pickup.pickup_person_phone or ""

    # If not in pickup object, try to get from timeline events
    if not observed_phone:
        for event in observed.timeline:
            if event.get("action") == "set_pickup_person":
                event_data = event.get("data", {})
                if isinstance(event_data, dict):
                    observed_phone = event_data.get("phone") or event_data.get("phoneNumber") or ""
                    break

    phone_passed = expected_phone == observed_phone if expected_phone else bool(observed_phone)
    add_check("pickup_person_phone", f"pickup person phone: {expected_phone}", observed_phone, phone_passed)

    return {"passed": len(failures) == 0, "failures": failures, "checks": checks}


def verify_payment_details(task: BasicPickupParams, observed: ApplicationState) -> Dict[str, Any]:
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
                expected_last4 = task.cc_number[-4:] if len(task.cc_number) >= 4 else ""
                observed_last4 = (
                    observed.payment.card_number_masked[-4:] if len(observed.payment.card_number_masked) >= 4 else ""
                )
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

        # Verify CCExpires from task against observed
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
                expected_last4 = task.cc_number[-4:] if len(task.cc_number) >= 4 else ""
                observed_last4 = (
                    observed.payment.card_number_masked[-4:] if len(observed.payment.card_number_masked) >= 4 else ""
                )
                passed = expected_last4 == observed_last4
                add_check(
                    "card_number", f"user's card ending in {expected_last4}", f"ending in {observed_last4}", passed
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


def verify_billing_address(task: BasicPickupParams, observed: ApplicationState) -> Dict[str, Any]:
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
            "streetAddress": observed.address.street_address,
            "city": observed.address.city,
            "state": observed.address.state,
            "zipCode": observed.address.zip_code,
        }

    # Verify street address
    if user_address["street"] and observed_billing.get("streetAddress"):
        expected_street = user_address["street"]
        observed_street = observed_billing["streetAddress"]
        passed = expected_street == observed_street
        add_check(
            "billing_street", f"user's street: {user_address['street']}", observed_billing["streetAddress"], passed
        )

    # Verify city
    if user_address["city"] and observed_billing.get("city"):
        expected_city = user_address["city"]
        observed_city = observed_billing["city"]
        passed = expected_city == observed_city
        add_check("billing_city", f"user's city: {user_address['city']}", observed_billing["city"], passed)

    # Verify state
    if user_address["state"] and observed_billing.get("state"):
        expected_state = user_address["state"]
        observed_state = observed_billing["state"]
        passed = expected_state == observed_state
        add_check("billing_state", f"user's state: {user_address['state']}", observed_billing["state"], passed)

    # Verify zip code
    if user_address["zip"] and observed_billing.get("zipCode"):
        expected_zip = user_address["zip"]
        observed_zip = observed_billing["zipCode"]
        passed = expected_zip == observed_zip
        add_check("billing_zip", f"user's zip: {user_address['zip']}", observed_billing["zipCode"], passed)

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
            expected = agent_output.order_number
            observed_val = observed.confirmation.order_number
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


def verify_basic_pickup_complete(
    task: BasicPickupParams,
    observed: ApplicationState,
    agent_output: Optional[AgentOutput] = None,
    timed_out: bool = False,
) -> VerificationResult:
    """Main verification function for basic pickup completion."""
    pickup_method_result = verify_pickup_method_selected(observed, task.pickup_method)
    pickup_person_result = verify_pickup_person(task, observed)
    payment_details_result = verify_payment_details(task, observed)
    billing_address_result = verify_billing_address(task, observed)
    order_completion_result = verify_order_completion(observed, agent_output)

    # Collect all failures
    all_failures = []
    if not pickup_method_result["passed"]:
        all_failures.extend(pickup_method_result["failures"])
    if not pickup_person_result["passed"]:
        all_failures.extend(pickup_person_result["failures"])
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
        pickup_method_result["passed"]
        and pickup_person_result["passed"]
        and payment_details_result["passed"]
        and billing_address_result["passed"]
        and order_completion_result["passed"]
        and not timed_out
    )

    details = {
        "pickup_method": pickup_method_result,
        "pickup_person": pickup_person_result,
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
    pickup_data = data.get("pickup", {})
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

    # Handle pickup person data - sim stores in pickupPersons[] array, need to normalize to pickup.person
    pickup_person_first_name = pickup_data.get("pickup_person_first_name")
    pickup_person_last_name = pickup_data.get("pickup_person_last_name")
    pickup_person_email = pickup_data.get("pickup_person_email")
    pickup_person_phone = pickup_data.get("pickup_person_phone")

    # Fallback: if pickup.person data is missing, try to get from pickupPersons[] array
    if not pickup_person_first_name and not pickup_person_last_name:
        pickup_persons = data.get("pickupPersons", [])
        if isinstance(pickup_persons, list) and len(pickup_persons) > 0:
            first_person = pickup_persons[0]
            if isinstance(first_person, dict):
                pickup_person_first_name = first_person.get("firstName")
                pickup_person_last_name = first_person.get("lastName")
                pickup_person_email = first_person.get("emailAddress")
                pickup_person_phone = first_person.get("phoneNumber")

    # Fallback: if still missing, try to get from timeline events
    if not pickup_person_first_name and not pickup_person_last_name:
        for event in timeline_data:
            if isinstance(event, dict) and event.get("action") == "set_pickup_person":
                event_data = event.get("data", {})
                if isinstance(event_data, dict):
                    pickup_person_first_name = event_data.get("first_name") or event_data.get("firstName")
                    pickup_person_last_name = event_data.get("last_name") or event_data.get("lastName")
                    pickup_person_email = event_data.get("email") or event_data.get("emailAddress")
                    pickup_person_phone = event_data.get("phone") or event_data.get("phoneNumber")
                    break

    # Final fallback: if still missing, try to get from user info (for default pickup person)
    if not pickup_person_first_name and not pickup_person_last_name:
        user_info = data.get("user_info", {})
        if user_info:
            pickup_person_first_name = user_info.get("GivenName")
            pickup_person_last_name = user_info.get("Surname")
            pickup_person_email = user_info.get("EmailAddress")
            pickup_person_phone = user_info.get("TelephoneNumber")

    # Handle alternate pickup person
    alternate_pickup_person = None
    alternate_data = pickup_data.get("alternate_pickup_person")
    if alternate_data and isinstance(alternate_data, dict):
        alternate_pickup_person = AlternatePickupPerson(
            first_name=alternate_data.get("first_name"),
            last_name=alternate_data.get("last_name"),
            email=alternate_data.get("email"),
        )

    # Determine pickup method - check timeline events if not in pickup_data
    pickup_method = (
        pickup_data.get("pickup_method") or shipping_info.get("pickup_method") or shipping_info.get("delivery_method")
    )

    # If no pickup method found, leave as None - verification function will default to curbside
    # if pickup is selected but no explicit method is found in global state

    pickup_info = PickupInfo(
        store_name=pickup_data.get("store_name"),
        store_address=pickup_data.get("store_address"),
        pickup_person_first_name=pickup_person_first_name,
        pickup_person_last_name=pickup_person_last_name,
        pickup_person_email=pickup_person_email,
        pickup_person_phone=pickup_person_phone,
        alternate_pickup_person=alternate_pickup_person,
        pickup_date=pickup_data.get("pickup_date"),
        pickup_time=pickup_data.get("pickup_time"),
        pickup_method=pickup_method,
    )

    # Handle confirmation data - can be bool or dict
    if isinstance(confirmation_data, bool):
        confirmation = ConfirmationInfo(reached=confirmation_data)
    else:
        confirmation = ConfirmationInfo(
            reached=confirmation_data.get("reached", False),
            order_number=confirmation_data.get("order_number"),
            confirmation_text_present=confirmation_data.get("confirmation_text_present", False),
        )

    # Extract cart data
    cart_items = []
    if cart_data and cart_data.get("items"):
        for item_data in cart_data["items"]:
            cart_item = CartItem(
                id=item_data.get("id"), title=item_data.get("title"), fulfillment=item_data.get("fulfillment", {})
            )
            cart_items.append(cart_item)

    cart_info = CartInfo(items=cart_items) if cart_items else None

    return ApplicationState(
        url=data.get("url"),
        contact=contact,
        address=address,
        payment=payment,
        pickup=pickup_info,
        confirmation=confirmation,
        timeline=timeline_data,
        cart=cart_info,
    )


def create_basic_pickup_params_from_task(task: Dict[str, Any]) -> BasicPickupParams:
    """Create BasicPickupParams from task dictionary."""
    values = task.get("values", {})
    user_info = values.get("user_info", {})

    # Extract parameters from task description
    task_description = task.get("task", "")

    # Parse pickup person name from task description
    pickup_person_name = "John Doe"  # default
    if "set the pickup person to john doe" in task_description.lower():
        pickup_person_name = "John Doe"
    elif "set the pickup person to michelle feller" in task_description.lower():
        pickup_person_name = "Michelle Feller"
    elif "pickup person" in task_description.lower():
        # Extract pickup person name if specified differently
        match = re.search(r"pickup person to ([^.]+?)(?:\.|$)", task_description)
        if match:
            pickup_person_name = match.group(1).strip().title()

    # Parse pickup method from task description
    pickup_method = "curbside"  # default
    if "curbside" in task_description.lower():
        pickup_method = "curbside"
    elif "in-store" in task_description.lower() or "in store" in task_description.lower():
        pickup_method = "in-store"

    # Parse payment method type
    payment_method_type = "default"  # default for most pickup tasks
    if "use the default payment method" in task_description.lower():
        payment_method_type = "default"
    elif (
        "card ending in" in task_description.lower()
        or "use user's card" in task_description.lower()
        or "use john doe's payment info" in task_description.lower()
        or "payment info" in task_description.lower()
    ):
        payment_method_type = "user"

    return BasicPickupParams(
        payment_method_type=payment_method_type,
        pickup_person_name=pickup_person_name,
        pickup_method=pickup_method,
        task_description=task.get("task", ""),
        expected_store_name=None,  # Allow any store selection
        # Payment details from user_info
        cc_type=user_info.get("CCType"),
        cc_number=user_info.get("CCNumber"),
        cvv2=user_info.get("CVV2"),
        cc_expires=user_info.get("CCExpires"),
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
    )


# =========================
# Evaluator class
# =========================


class InputDict(TypedDict, total=False):
    page: "Page"


class FinalResult(BaseModel):
    """Result model for basic pickup task."""

    score: float
    reasoning: str
    current_state: dict[str, Any]
    ground_truth: Any


class HalluminateGoodBuyBasicPickup(BaseMetric):
    """Evaluator that runs verifier for basic pickup task."""

    def __init__(self, task: str, values: dict[str, Any], user_metadata: UserMetadata) -> None:
        super().__init__()
        self.task = task
        self.values = values
        self.user_metadata = user_metadata
        # Create ground truth from task values (verifier expects {"task": task, "values": values})
        self._ground_truth = create_basic_pickup_params_from_task({"task": task, "values": values})
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
        result = verify_basic_pickup_complete(self._ground_truth, state_obj)

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


# =========================
# Task config generation
# =========================


def _render_values(values: dict[str, Any], timestamp: int) -> dict[str, Any]:
    """Render values for basic_pickup tasks.

    Pickup tasks don't use date formulas.
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
    values = _render_values(values, timestamp)

    # Render the task (replace placeholders with formatted values)
    task = _render_task(task, values)

    # Create the eval config
    eval_target = get_import_path(HalluminateGoodBuyBasicPickup)
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
        "task_id": "halluminate_bench/goodbuy/basic_pickup/1",
        "task_generation_config_json": json.dumps(
            {
                "_target_": "halluminate_bench.goodbuy.basic_pickup.generate_task_config",
                "url": "https://good-buy.halluminate.ai/?apiKey={HALLUMINATE_API_KEY}",
                "task": (
                    "I want to pick this item up in-store. Set the pickup person to Violet Ritchie. "
                    "Use Violet Ritchie's payment info. Complete the order and stop once you reach "
                    "the confirmation page. Use guest checkout. User info: {user_info}."
                ),
                "values": {
                    "pickup_method": "in-store",
                    "pickup_person": "Violet Ritchie",
                    "payment_method": "user",
                    "user_info": {
                        "Title": "Ms.",
                        "GivenName": "Violet",
                        "MiddleInitial": "R",
                        "Surname": "Ritchie",
                        "StreetAddress": "466 Vine Street",
                        "City": "Fox Lake",
                        "State": "IL",
                        "StateFull": "Illinois",
                        "ZipCode": "60020",
                        "Country": "US",
                        "CountryFull": "United States",
                        "EmailAddress": "VioletRRitchie@dayrep.com",
                        "TelephoneNumber": "847-587-9477",
                        "TelephoneCountryCode": "1",
                        "Birthday": "7/27/1936",
                        "Age": "88",
                        "CCType": "MasterCard",
                        "CCNumber": "5200007351577350",
                        "CVV2": "978",
                        "CCExpires": "5/2027",
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
        "l2_category": "basic_pickup",
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
