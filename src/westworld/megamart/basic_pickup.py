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

    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone_number: Optional[str] = None
    street_address: Optional[str] = None
    apt_suite: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    country: Optional[str] = None


@dataclass
class PaymentInfo:
    """Payment information for the pickup."""

    payment_method: Optional[str] = None  # "default" or "user_card"
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone_number: Optional[str] = None
    card_type: Optional[str] = None
    card_number: Optional[str] = None  # full digits when available
    card_number_masked: Optional[str] = None  # e.g., **** **** **** 1234
    expiry: Optional[str] = None
    expiry_month: Optional[str] = None  # Separate month field
    expiry_year: Optional[str] = None  # Separate year field
    cvv_present: Optional[bool] = None
    cvv: Optional[str] = None
    used_default: Optional[bool] = None
    billing_address: Optional[AddressInfo] = None


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
    alternate_pickup_person: Optional[AlternatePickupPerson] = None
    pickup_date: Optional[str] = None  # actual date in YYYY-MM-DD format
    pickup_time: Optional[str] = None  # "after 11am", "between 2pm-4pm", etc.
    bag_option: Optional[str] = None  # "want-bags" or "own-bags"


@dataclass
class DefaultPickupInfo:
    """Default pickup person information from simulator."""

    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None


@dataclass
class ConfirmationInfo:
    """Confirmation information for the pickup."""

    reached: bool = False
    order_number: Optional[str] = None
    confirmation_text_present: bool = False


@dataclass
class ApplicationState:
    """Current state of the pickup application."""

    contact: ContactInfo
    address: AddressInfo
    payment: PaymentInfo
    pickup: PickupInfo
    confirmation: ConfirmationInfo
    timeline: List[Dict[str, Any]]
    url: Optional[str] = None
    default_address: Optional[AddressInfo] = None  # Default address from simulator
    default_payment: Optional[PaymentInfo] = None  # Default payment from simulator
    default_pickup: Optional[DefaultPickupInfo] = None  # Default pickup person from simulator


@dataclass
class BasicPickupParams:
    """Expected pickup parameters from the basic pickup task."""

    payment_method_type: str  # "default" or "user"
    bag_option: str  # "own-bags" or "want-bags"
    pickup_person_type: str  # "default" or "user"
    pickup_person_name: Optional[str] = None  # Full name when using user-specific pickup person
    task_description: str = ""  # Original task description for conditional verification
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


def verify_pickup_method_selected(observed: ApplicationState, require_store: bool = False) -> Dict[str, Any]:
    """Verify pickup method selection and optionally store selection."""
    failures = []
    checks = []

    def add_check(name: str, expected: str, observed_val: str, passed: bool):
        checks.append({"check": name, "expected": expected, "observed": observed_val, "passed": passed})
        if not passed:
            failures.append(name)

    # Check if pickup method was selected by looking at timeline
    pickup_selected = False
    for event in observed.timeline:
        if (
            event.get("action") == "changed_delivery_method"
            and event.get("data", {}).get("delivery_method") == "pickup"
        ):
            pickup_selected = True
            break

    # Detect store presence
    store_name = (observed.pickup.store_name or "").strip() if observed.pickup else ""
    store_selected = bool(store_name)

    # Composite check based on mode
    if require_store:
        composite_pass = pickup_selected and store_selected
        expected_text = "pickup method and store selected"
        observed_text = f"pickup: {'selected' if pickup_selected else 'not selected'}, store: {store_name or 'None'}"
    else:
        composite_pass = pickup_selected
        expected_text = "pickup method selected"
        observed_text = f"pickup: {'selected' if pickup_selected else 'not selected'}"

    add_check("pickup_method_selected", expected_text, observed_text, composite_pass)

    return {"passed": len(failures) == 0, "failures": failures, "checks": checks}


def verify_pickup_person(task: BasicPickupParams, observed: ApplicationState) -> Dict[str, Any]:
    """Verify pickup person information matches expected values."""
    failures = []
    checks = []

    def add_check(name: str, expected: str, observed_val: str, passed: bool):
        checks.append({"check": name, "expected": expected, "observed": observed_val, "passed": passed})
        if not passed:
            failures.append(name)

    # Check if task requires default pickup person
    if task.pickup_person_type == "default":
        # Verify that actual pickup person matches default pickup person
        if not observed.default_pickup:
            add_check(
                "default_pickup_missing",
                "default_pickup_info present in state",
                "None",
                False,
            )
        else:
            # Compare ALL pickup person fields with defaults
            # Check first_name
            expected = normalize_string_for_comparison(observed.default_pickup.first_name or "")
            observed_val = normalize_string_for_comparison(observed.pickup.pickup_person_first_name or "")
            passed = expected == observed_val
            add_check(
                "default_pickup_first_name_match",
                observed.default_pickup.first_name or "(empty)",
                observed.pickup.pickup_person_first_name or "(empty)",
                passed,
            )

            # Check last_name
            expected = normalize_string_for_comparison(observed.default_pickup.last_name or "")
            observed_val = normalize_string_for_comparison(observed.pickup.pickup_person_last_name or "")
            passed = expected == observed_val
            add_check(
                "default_pickup_last_name_match",
                observed.default_pickup.last_name or "(empty)",
                observed.pickup.pickup_person_last_name or "(empty)",
                passed,
            )

            # Check email
            expected = normalize_string_for_comparison(observed.default_pickup.email or "")
            observed_val = normalize_string_for_comparison(observed.pickup.pickup_person_email or "")
            passed = expected == observed_val
            add_check(
                "default_pickup_email_match",
                observed.default_pickup.email or "(empty)",
                observed.pickup.pickup_person_email or "(empty)",
                passed,
            )
    else:
        # Verify pickup person name matches task requirement (user-specific)
        if task.pickup_person_name:
            name_parts = task.pickup_person_name.split()
            expected_first = normalize_string_for_comparison(name_parts[0])
            expected_last = normalize_string_for_comparison(" ".join(name_parts[1:])) if len(name_parts) > 1 else ""

            observed_first = normalize_string_for_comparison(observed.pickup.pickup_person_first_name or "")
            observed_last = normalize_string_for_comparison(observed.pickup.pickup_person_last_name or "")

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

        # Verify pickup person email matches user_info
        if task.email:
            expected_email = normalize_string_for_comparison(task.email)
            observed_email = normalize_string_for_comparison(observed.pickup.pickup_person_email or "")

            if not observed.pickup.pickup_person_email:
                add_check("pickup_person_email", f"pickup person email: {task.email}", "None", False)
            else:
                passed = expected_email == observed_email
                add_check(
                    "pickup_person_email",
                    f"pickup person email: {task.email}",
                    observed.pickup.pickup_person_email,
                    passed,
                )
        else:
            # No email in user_info, just verify email is present
            if observed.pickup.pickup_person_email:
                add_check(
                    "pickup_person_email", "pickup person email present", observed.pickup.pickup_person_email, True
                )
            else:
                add_check("pickup_person_email", "pickup person email present", "None", False)

    return {"passed": len(failures) == 0, "failures": failures, "checks": checks}


def verify_bag_option(task: BasicPickupParams, observed: ApplicationState) -> Dict[str, Any]:
    """Verify bag option matches task requirement."""
    failures = []
    checks = []

    def add_check(name: str, expected: str, observed_val: str, passed: bool):
        checks.append({"check": name, "expected": expected, "observed": observed_val, "passed": passed})
        if not passed:
            failures.append(name)

    # Verify bag option matches task requirement
    if task.bag_option:
        expected_bag_option = task.bag_option
        observed_bag_option = observed.pickup.bag_option or ""
        passed = expected_bag_option == observed_bag_option
        add_check("bag_option", f"bag option: {expected_bag_option}", observed_bag_option, passed)

    return {"passed": len(failures) == 0, "failures": failures, "checks": checks}


def verify_payment_details(task: BasicPickupParams, observed: ApplicationState) -> Dict[str, Any]:
    """Verify payment details match task requirements."""
    failures = []
    checks = []

    def add_check(name: str, expected: str, observed_val: str, passed: bool):
        checks.append({"check": name, "expected": expected, "observed": observed_val, "passed": passed})
        if not passed:
            failures.append(name)

    # Check payment method type
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
            # Compare ALL payment fields with defaults - must check every field
            # Check first_name
            expected = normalize_string_for_comparison(observed.default_payment.first_name or "")
            observed_val = normalize_string_for_comparison(observed.payment.first_name or "")
            passed = expected == observed_val
            add_check(
                "default_payment_first_name_match",
                observed.default_payment.first_name or "(empty)",
                observed.payment.first_name or "(empty)",
                passed,
            )

            # Check last_name
            expected = normalize_string_for_comparison(observed.default_payment.last_name or "")
            observed_val = normalize_string_for_comparison(observed.payment.last_name or "")
            passed = expected == observed_val
            add_check(
                "default_payment_last_name_match",
                observed.default_payment.last_name or "(empty)",
                observed.payment.last_name or "(empty)",
                passed,
            )

            # Check phone_number
            expected = re.sub(r"\D", "", observed.default_payment.phone_number or "")
            observed_val = re.sub(r"\D", "", observed.payment.phone_number or "")
            passed = expected == observed_val
            add_check(
                "default_payment_phone_match",
                observed.default_payment.phone_number or "(empty)",
                observed.payment.phone_number or "(empty)",
                passed,
            )

            # Check card_number
            expected = re.sub(r"\D", "", observed.default_payment.card_number or "")
            observed_val = re.sub(r"\D", "", observed.payment.card_number or "")
            passed = expected == observed_val
            add_check(
                "default_card_number_match",
                observed.default_payment.card_number or "(empty)",
                observed.payment.card_number or "(empty)",
                passed,
            )

            # Check expiry_month
            expected = normalize_string_for_comparison(observed.default_payment.expiry_month or "")
            observed_val = normalize_string_for_comparison(observed.payment.expiry_month or "")
            passed = expected == observed_val
            add_check(
                "default_expiry_month_match",
                observed.default_payment.expiry_month or "(empty)",
                observed.payment.expiry_month or "(empty)",
                passed,
            )

            # Check expiry_year
            expected = normalize_string_for_comparison(observed.default_payment.expiry_year or "")
            observed_val = normalize_string_for_comparison(observed.payment.expiry_year or "")
            passed = expected == observed_val
            add_check(
                "default_expiry_year_match",
                observed.default_payment.expiry_year or "(empty)",
                observed.payment.expiry_year or "(empty)",
                passed,
            )

            # Check cvv
            expected = observed.default_payment.cvv or ""
            observed_val = observed.payment.cvv or ""
            passed = expected == observed_val
            add_check(
                "default_cvv_match",
                observed.default_payment.cvv or "(empty)",
                observed.payment.cvv or "(empty)",
                passed,
            )

            # Check billing address - always check even if None
            default_billing = observed.default_payment.billing_address
            actual_billing = observed.payment.billing_address

            # If both are None, that's a match. If only one is None, that's a mismatch.
            if default_billing is None and actual_billing is None:
                add_check("default_billing_address_present", "None", "None", True)
            elif default_billing is None or actual_billing is None:
                add_check(
                    "default_billing_address_present",
                    "present" if default_billing else "None",
                    "present" if actual_billing else "None",
                    False,
                )
            else:
                # Both present, check all fields
                expected = normalize_string_for_comparison(default_billing.street_address or "")
                observed_val = normalize_string_for_comparison(actual_billing.street_address or "")
                passed = expected == observed_val
                add_check(
                    "default_billing_street_match",
                    default_billing.street_address or "(empty)",
                    actual_billing.street_address or "(empty)",
                    passed,
                )

                expected = normalize_string_for_comparison(default_billing.apt_suite or "")
                observed_val = normalize_string_for_comparison(actual_billing.apt_suite or "")
                passed = expected == observed_val
                add_check(
                    "default_billing_apt_suite_match",
                    default_billing.apt_suite or "(empty)",
                    actual_billing.apt_suite or "(empty)",
                    passed,
                )

                expected = normalize_string_for_comparison(default_billing.city or "")
                observed_val = normalize_string_for_comparison(actual_billing.city or "")
                passed = expected == observed_val
                add_check(
                    "default_billing_city_match",
                    default_billing.city or "(empty)",
                    actual_billing.city or "(empty)",
                    passed,
                )

                expected = normalize_string_for_comparison(default_billing.state or "")
                observed_val = normalize_string_for_comparison(actual_billing.state or "")
                passed = expected == observed_val
                add_check(
                    "default_billing_state_match",
                    default_billing.state or "(empty)",
                    actual_billing.state or "(empty)",
                    passed,
                )

                expected = normalize_string_for_comparison(default_billing.zip_code or "")[:5]
                observed_val = normalize_string_for_comparison(actual_billing.zip_code or "")[:5]
                passed = expected == observed_val
                add_check(
                    "default_billing_zip_match",
                    default_billing.zip_code or "(empty)",
                    actual_billing.zip_code or "(empty)",
                    passed,
                )

    else:
        # For user-specified payment, verify the specific card details match user's card
        if task.cc_number and observed.payment.card_number:
            expected = re.sub(r"\D", "", task.cc_number)
            observed_val = re.sub(r"\D", "", observed.payment.card_number or "")
            passed = expected == observed_val
            add_check("card_number_match", expected, observed_val, passed)

            # Verify card type if available
            if task.cc_type and observed.payment.card_type:
                expected_type = normalize_string_for_comparison(task.cc_type)
                observed_type = normalize_string_for_comparison(observed.payment.card_type)
                passed = expected_type == observed_type
                add_check("card_type", f"card type: {task.cc_type}", observed.payment.card_type, passed)

            # Verify expiry if available
            if task.cc_expires and observed.payment.expiry:
                expected_expiry = normalize_expiry(task.cc_expires)
                observed_expiry = normalize_expiry(observed.payment.expiry)
                passed = expected_expiry == observed_expiry
                add_check("card_expiry", f"card expiry: {task.cc_expires}", observed.payment.expiry, passed)

    return {"passed": len(failures) == 0, "failures": failures, "checks": checks}


def verify_billing_address(task: BasicPickupParams, observed: ApplicationState) -> Dict[str, Any]:
    """Verify billing address matches expected address based on payment method type."""
    failures = []
    checks = []

    def add_check(name: str, expected: str, observed_val: str, passed: bool):
        checks.append({"check": name, "expected": expected, "observed": observed_val, "passed": passed})
        if not passed:
            failures.append(name)

    # Get observed billing address from payment info
    observed_billing = observed.payment.billing_address

    # Verify billing address based on payment method type
    if task.payment_method_type == "default":
        # Verify billing address matches default payment info
        default_billing = observed.default_payment.billing_address if observed.default_payment else None

        # If both are None, that's acceptable - no billing address required
        if default_billing is None and observed_billing is None:
            add_check("billing_address_match", "None (no billing address)", "None", True)
            return {"passed": True, "failures": failures, "checks": checks}

        # If only one is None, that's a mismatch
        if default_billing is None or observed_billing is None:
            add_check(
                "billing_address_match",
                "present" if default_billing else "None",
                "present" if observed_billing else "None",
                False,
            )
            return {"passed": False, "failures": failures, "checks": checks}

        # Both present, compare all billing address fields with defaults
        expected = normalize_string_for_comparison(default_billing.street_address or "")
        observed_val = normalize_string_for_comparison(observed_billing.street_address or "")
        passed = expected == observed_val
        add_check(
            "billing_street_match",
            default_billing.street_address or "(empty)",
            observed_billing.street_address or "(empty)",
            passed,
        )

        expected = normalize_string_for_comparison(default_billing.city or "")
        observed_val = normalize_string_for_comparison(observed_billing.city or "")
        passed = expected == observed_val
        add_check(
            "billing_city_match",
            default_billing.city or "(empty)",
            observed_billing.city or "(empty)",
            passed,
        )

        expected = normalize_string_for_comparison(default_billing.state or "")
        observed_val = normalize_string_for_comparison(observed_billing.state or "")
        passed = expected == observed_val
        add_check(
            "billing_state_match",
            default_billing.state or "(empty)",
            observed_billing.state or "(empty)",
            passed,
        )

        expected = normalize_string_for_comparison(default_billing.zip_code or "")[:5]
        observed_val = normalize_string_for_comparison(observed_billing.zip_code or "")[:5]
        passed = expected == observed_val
        add_check(
            "billing_zip_match",
            default_billing.zip_code or "(empty)",
            observed_billing.zip_code or "(empty)",
            passed,
        )

    else:
        # payment_method_type is "user" - verify billing address matches user's address from task
        # For user payment method, billing address is required
        if not observed_billing:
            add_check("billing_address_present", "billing address present", "None", False)
            return {"passed": False, "failures": failures, "checks": checks}

        expected_street = normalize_string_for_comparison(task.street_address or "")
        observed_street = normalize_string_for_comparison(observed_billing.street_address or "")
        passed = expected_street == observed_street
        add_check(
            "billing_street_match",
            task.street_address or "(empty)",
            observed_billing.street_address or "(empty)",
            passed,
        )

        expected_city = normalize_string_for_comparison(task.city or "")
        observed_city = normalize_string_for_comparison(observed_billing.city or "")
        passed = expected_city == observed_city
        add_check(
            "billing_city_match",
            task.city or "(empty)",
            observed_billing.city or "(empty)",
            passed,
        )

        expected_state = normalize_string_for_comparison(task.state or "")
        observed_state = normalize_string_for_comparison(observed_billing.state or "")
        passed = expected_state == observed_state
        add_check(
            "billing_state_match",
            task.state or "(empty)",
            observed_billing.state or "(empty)",
            passed,
        )

        expected_zip = task.zip_code or ""
        observed_zip = observed_billing.zip_code or ""
        # Compare first 5 digits only
        passed = expected_zip[:5] == observed_zip[:5]
        add_check(
            "billing_zip_match",
            task.zip_code or "(empty)",
            observed_billing.zip_code or "(empty)",
            passed,
        )

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


def verify_basic_pickup_complete(
    task: BasicPickupParams,
    observed: ApplicationState,
    agent_output: Optional[AgentOutput] = None,
    timed_out: bool = False,
    require_store_for_pickup_method: bool = False,
) -> VerificationResult:
    """Main verification function for basic pickup completion."""
    pickup_method_result = verify_pickup_method_selected(observed, require_store=require_store_for_pickup_method)
    pickup_person_result = verify_pickup_person(task, observed)
    bag_option_result = verify_bag_option(task, observed)
    payment_details_result = verify_payment_details(task, observed)
    billing_address_result = verify_billing_address(task, observed)
    order_completion_result = verify_order_completion(observed, agent_output)

    # Collect all failures
    all_failures = []
    if not pickup_method_result["passed"]:
        all_failures.extend(pickup_method_result["failures"])
    if not pickup_person_result["passed"]:
        all_failures.extend(pickup_person_result["failures"])
    if not bag_option_result["passed"]:
        all_failures.extend(bag_option_result["failures"])
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
        and bag_option_result["passed"]
        and payment_details_result["passed"]
        and billing_address_result["passed"]
        and order_completion_result["passed"]
        and not timed_out
    )

    details = {
        "pickup_method": pickup_method_result,
        "pickup_person": pickup_person_result,
        "bag_option": bag_option_result,
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
    default_shipping_info = data.get("default_shipping_info", {})
    default_payment_info = data.get("default_payment_info", {})
    pickup_data = data.get("pickup", {})
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
        first_name=shipping_info.get("firstName"),
        last_name=shipping_info.get("lastName"),
        phone_number=shipping_info.get("phoneNumber"),
        street_address=shipping_info.get("streetAddress"),
        apt_suite=shipping_info.get("aptSuite"),
        city=shipping_info.get("city"),
        state=shipping_info.get("state"),
        zip_code=shipping_info.get("zipCode"),
        country=None,  # Not available in new format
    )

    # Extract default address info from default_shipping_info
    default_address = None
    if default_shipping_info:
        default_address = AddressInfo(
            first_name=default_shipping_info.get("firstName"),
            last_name=default_shipping_info.get("lastName"),
            phone_number=default_shipping_info.get("phoneNumber"),
            street_address=default_shipping_info.get("streetAddress"),
            apt_suite=default_shipping_info.get("aptSuite"),
            city=default_shipping_info.get("city"),
            state=default_shipping_info.get("state"),
            zip_code=default_shipping_info.get("zipCode"),
            country=None,
        )

    # Extract payment info from payment_info
    billing_address_dict = payment_info.get("billingAddress", {})
    billing_address = (
        AddressInfo(
            first_name=None,  # Not in billing address
            last_name=None,
            phone_number=None,
            street_address=billing_address_dict.get("streetAddress"),
            apt_suite=billing_address_dict.get("aptSuite"),
            city=billing_address_dict.get("city"),
            state=billing_address_dict.get("state"),
            zip_code=billing_address_dict.get("zipCode"),
        )
        if billing_address_dict
        else None
    )

    payment = PaymentInfo(
        payment_method="user_card" if payment_info.get("cardNumber") else "default",
        first_name=payment_info.get("firstName"),
        last_name=payment_info.get("lastName"),
        phone_number=payment_info.get("phoneNumber"),
        card_type=None,  # Not available in new format
        card_number=payment_info.get("cardNumber"),
        card_number_masked=None,  # Not available in new format
        expiry=f"{payment_info.get('expiryMonth', '')}/{payment_info.get('expiryYear', '')}"
        if payment_info.get("expiryMonth") and payment_info.get("expiryYear")
        else None,
        expiry_month=payment_info.get("expiryMonth"),
        expiry_year=payment_info.get("expiryYear"),
        cvv_present=bool(payment_info.get("cvv")),
        cvv=payment_info.get("cvv"),
        used_default=payment_info.get("setAsDefault"),
        billing_address=billing_address,
    )

    # Extract default payment info from default_payment_info
    default_payment = None
    if default_payment_info:
        default_billing_address_dict = default_payment_info.get("billingAddress", {})
        default_billing_address = (
            AddressInfo(
                first_name=None,
                last_name=None,
                phone_number=None,
                street_address=default_billing_address_dict.get("streetAddress"),
                apt_suite=default_billing_address_dict.get("aptSuite"),
                city=default_billing_address_dict.get("city"),
                state=default_billing_address_dict.get("state"),
                zip_code=default_billing_address_dict.get("zipCode"),
            )
            if default_billing_address_dict
            else None
        )

        default_payment = PaymentInfo(
            payment_method="default",
            first_name=default_payment_info.get("firstName"),
            last_name=default_payment_info.get("lastName"),
            phone_number=default_payment_info.get("phoneNumber"),
            card_type=None,
            card_number=default_payment_info.get("cardNumber"),
            card_number_masked=None,
            expiry=f"{default_payment_info.get('expiryMonth', '')}/{default_payment_info.get('expiryYear', '')}"
            if default_payment_info.get("expiryMonth") and default_payment_info.get("expiryYear")
            else None,
            expiry_month=default_payment_info.get("expiryMonth"),
            expiry_year=default_payment_info.get("expiryYear"),
            cvv_present=bool(default_payment_info.get("cvv")),
            cvv=default_payment_info.get("cvv"),
            used_default=None,
            billing_address=default_billing_address,
        )

    # Extract default pickup info from default_pickup
    default_pickup = None
    default_pickup_info = data.get("default_pickup", {})
    if default_pickup_info:
        default_pickup = DefaultPickupInfo(
            first_name=default_pickup_info.get("pickup_person_first_name"),
            last_name=default_pickup_info.get("pickup_person_last_name"),
            email=default_pickup_info.get("pickup_person_email"),
        )

    # Handle alternate pickup person
    alternate_pickup_person = None
    alternate_data = pickup_data.get("alternate_pickup_person")
    if alternate_data and isinstance(alternate_data, dict):
        alternate_pickup_person = AlternatePickupPerson(
            first_name=alternate_data.get("first_name"),
            last_name=alternate_data.get("last_name"),
            email=alternate_data.get("email"),
        )

    pickup_info = PickupInfo(
        store_name=pickup_data.get("store_name"),
        store_address=pickup_data.get("store_address"),
        pickup_person_first_name=pickup_data.get("pickup_person_first_name"),
        pickup_person_last_name=pickup_data.get("pickup_person_last_name"),
        pickup_person_email=pickup_data.get("pickup_person_email"),
        alternate_pickup_person=alternate_pickup_person,
        pickup_date=pickup_data.get("pickup_date"),
        pickup_time=pickup_data.get("pickup_time"),
        bag_option=pickup_data.get("bag_option"),
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
        pickup=pickup_info,
        confirmation=confirmation,
        timeline=timeline_data,
        default_address=default_address,
        default_payment=default_payment,
        default_pickup=default_pickup,
    )


def create_basic_pickup_params_from_task(task: Dict[str, Any]) -> BasicPickupParams:
    """Create BasicPickupParams from task dictionary."""
    values = task.get("values", {})
    user_info = values.get("user_info", {})

    # Get parameters directly from values (preferred) or fall back to parsing task description
    pickup_person_value = values.get("pickup_person")
    bag_option = values.get("bag_option")
    payment_method_type = values.get("payment_method")

    # Determine pickup person type and name
    if pickup_person_value == "default":
        pickup_person_type = "default"
        pickup_person_name = None
    elif pickup_person_value:
        pickup_person_type = "user"
        pickup_person_name = pickup_person_value
    else:
        # Fall back to parsing task description for backward compatibility
        task_description = task.get("task", "").lower()
        if "keep the primary pickup person as the current default" in task_description:
            pickup_person_type = "default"
            pickup_person_name = None
        else:
            pickup_person_type = "user"
            # Extract pickup person name from task description
            if "set the primary pickup person as john doe" in task_description:
                pickup_person_name = "John Doe"
            elif "set the primary pickup person as michelle feller" in task_description:
                pickup_person_name = "Michelle Feller"
            elif "pickup person as" in task_description:
                match = re.search(r"pickup person as ([^.?]+)", task_description)
                if match:
                    pickup_person_name = match.group(1).strip().title()
            else:
                pickup_person_name = "John Doe"  # fallback default

    # Fall back to parsing task description if bag_option or payment_method_type are not present
    if not bag_option or not payment_method_type:
        task_description = task.get("task", "").lower()

        if not bag_option:
            # Parse bag option from task description
            bag_option = "want-bags"  # default
            if "bring my own reusable bags" in task_description or "bring my own bags" in task_description:
                bag_option = "own-bags"
            elif "want my items in a bag" in task_description or "want bags" in task_description:
                bag_option = "want-bags"

        if not payment_method_type:
            # Parse payment method type
            payment_method_type = "default"  # default for most pickup tasks
            if "use the default payment method" in task_description:
                payment_method_type = "default"
            elif "card ending in" in task_description or "card" in task_description:
                payment_method_type = "user"

    return BasicPickupParams(
        payment_method_type=payment_method_type,
        bag_option=bag_option,
        pickup_person_type=pickup_person_type,
        pickup_person_name=pickup_person_name,
        task_description=task.get("task", ""),
        expected_store_name=None,  # Allow any store selection
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
    """Result model for Megamart basic pickup task."""

    score: float
    reasoning: str
    current_state: dict[str, Any]
    ground_truth: Any


class HalluminateMegamartBasicPickup(BaseMetric):
    """Evaluator that runs verifier for Megamart basic pickup task."""

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
        return await self._page.evaluate("() => window.__MEGAMART_EVAL_STATE__")


def _render_values(values: dict[str, Any], timestamp: int) -> dict[str, Any]:
    """Render values for basic_pickup tasks.

    For basic_pickup, we don't have date formulas, but we keep this function
    for consistency with other evaluators.
    """
    return dict(values)


def _render_task(task: str, values: dict[str, Any]) -> str:
    """Replace {pickup_person}, {bag_option}, {payment_method}, {user_info} placeholders.

    Example:
        Input task: "Can you {pickup_person}? I want {bag_option}..."
        Input values: {"pickup_person": "Ruth Barley", "bag_option": "want-bags", ...}
        Output: "Can you set the primary pickup person as Ruth Barley? I want my items in a bag..."
    """
    result = task

    # Replace {pickup_person}
    if "{pickup_person}" in result:
        pickup_person = values.get("pickup_person", "")
        if pickup_person == "default":
            result = result.replace("{pickup_person}", "keep the primary pickup person as the current default")
        else:
            result = result.replace("{pickup_person}", f"set the primary pickup person as {pickup_person}")

    # Replace {bag_option}
    if "{bag_option}" in result:
        bag_option = values.get("bag_option", "")
        if bag_option == "own-bags":
            # Handle "I want {bag_option}" -> "I want to bring my own reusable bags"
            if "I want {bag_option}" in result:
                result = result.replace("I want {bag_option}", "I want to bring my own reusable bags")
            else:
                result = result.replace("{bag_option}", "to bring my own reusable bags")
        elif bag_option == "want-bags":
            # Handle "I want {bag_option}" -> "I want my items in a bag"
            if "I want {bag_option}" in result:
                result = result.replace("I want {bag_option}", "I want my items in a bag")
            else:
                result = result.replace("{bag_option}", "my items in a bag")
        else:
            # Fallback
            result = result.replace("{bag_option}", bag_option)

    # Replace {payment_method}
    if "{payment_method}" in result:
        payment_method = values.get("payment_method", "")
        user_info = values.get("user_info", {})

        if payment_method == "default":
            result = result.replace("{payment_method}", "the default payment method")
        elif payment_method == "user":
            # Format: "{GivenName}'s card ending in {last4} and set the billing address to their address"
            given_name = user_info.get("GivenName", "")
            card_number = user_info.get("CCNumber", "")
            last_four = card_number[-4:] if card_number and len(card_number) >= 4 else "0000"
            payment_text = f"{given_name}'s card ending in {last_four} and set the billing address to their address"
            result = result.replace("{payment_method}", payment_text)
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
    3. Renders task text with formatted values
    4. Creates eval_config pointing to evaluator

    Args:
        url: Base URL with {HALLUMINATE_API_KEY} placeholder
        task: Task template with placeholders
        values: Task values (pickup_person, bag_option, payment_method, user_info)
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

    # Render the values (no date formulas for pickup, but keep for consistency)
    values = _render_values(values, timestamp)

    # Render the task (replace placeholders with formatted values)
    task = _render_task(task, values)

    # Create the eval config
    eval_target = get_import_path(HalluminateMegamartBasicPickup)
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
        "task_id": "halluminate_bench/megamart/basic_pickup/1",
        "task_generation_config_json": json.dumps(
            {
                "_target_": "halluminate_bench.megamart.basic_pickup.generate_task_config",
                "url": "https://megamart.halluminate.ai/?apiKey={HALLUMINATE_API_KEY}",
                "task": (
                    "I want to order this vase for Curbside Pickup. "
                    "Can you {pickup_person}? "
                    "I want {bag_option}. "
                    "For payment, use {payment_method}. "
                    "Your job is done when the order is successfully placed. "
                    "You may need to reference the following user info: {user_info}"
                ),
                "values": {
                    "pickup_person": "Ruth Barley",
                    "bag_option": "want-bags",
                    "payment_method": "user",
                    "user_info": {
                        "Title": "Mrs.",
                        "GivenName": "Ruth",
                        "MiddleInitial": "R",
                        "Surname": "Barley",
                        "StreetAddress": "3126 Stratford Court",
                        "City": "Morrisville",
                        "State": "NC",
                        "StateFull": "North Carolina",
                        "ZipCode": "27560",
                        "Country": "US",
                        "CountryFull": "United States",
                        "EmailAddress": "RuthRBarley@einrot.com",
                        "TelephoneNumber": "919-760-1737",
                        "TelephoneCountryCode": "1",
                        "CCType": "Visa",
                        "CCNumber": "4916618483369843",
                        "CVV2": "260",
                        "CCExpires": "10/2029",
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
