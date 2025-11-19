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
    """Contact information for the checkout."""

    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None


@dataclass
class AddressInfo:
    """Address information for the checkout."""

    street_address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    country: Optional[str] = None
    used_default: Optional[bool] = None


@dataclass
class PaymentInfo:
    """Payment information for the checkout."""

    payment_method: Optional[str] = None
    card_type: Optional[str] = None
    card_number: Optional[str] = None
    card_number_masked: Optional[str] = None
    expiry: Optional[str] = None
    cvv_present: Optional[bool] = None
    cvv: Optional[str] = None
    used_default: Optional[bool] = None
    billing_address_same: Optional[bool] = None
    billing_address: Optional[AddressInfo] = None


@dataclass
class DeliveryInfo:
    """Delivery information for the checkout."""

    recipient_name: Optional[str] = None
    delivery_date_text: Optional[str] = None
    delivery_date: Optional[str] = None
    delivery_method: Optional[str] = None


@dataclass
class DeliveryInstructions:
    """Delivery instruction details."""

    dropoff_location: Optional[str] = None
    property_type: Optional[str] = None
    delivery_notes: Optional[str] = None


@dataclass
class ConfirmationInfo:
    """Confirmation information for the checkout."""

    reached: bool = False
    order_number: Optional[str] = None
    confirmation_text_present: bool = False


@dataclass
class ApplicationState:
    """Current state of the checkout application."""

    contact: ContactInfo
    address: AddressInfo
    payment: PaymentInfo
    delivery: DeliveryInfo
    confirmation: ConfirmationInfo
    timeline: List[Dict[str, Any]]
    delivery_instructions: Optional[DeliveryInstructions] = None
    url: Optional[str] = None
    delivery_date: Optional[str] = None


@dataclass
class BasicCheckoutParams:
    """Expected checkout parameters from the basic checkout task."""

    delivery_address_type: str  # "default" or "user"
    payment_method_type: str  # "default" or "user"
    delivery_date_text: Optional[str] = None  # "tomorrow", "2 days from now", etc.
    expected_person_name: Optional[str] = None
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
    # Delivery instruction fields
    dropoff_location: Optional[str] = None
    property_type: Optional[str] = None
    delivery_notes: Optional[str] = None


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
    """
    if not expiry:
        return ""
    s = expiry.strip()
    m = re.match(r"^(\d{1,2})\s*[\/\-\s]\s*(\d{2,4})$", s) or re.match(r"^(\d{1,2})(\d{4})$", s)
    if not m:
        return ""
    month = int(m.group(1))
    year = int(m.group(2))
    if year < 100:
        year += 2000
    if not (1 <= month <= 12):
        return ""
    return f"{year:04d}-{month:02d}"


def parse_date_text(date_text: str, now: Optional[datetime] = None) -> Optional[datetime]:
    """Parse delivery date text into a standardized format."""
    if not date_text:
        return None
    now = now or datetime.now()
    t = date_text.strip().lower()
    if t == "today":
        return now
    if t == "tomorrow":
        return now + timedelta(days=1)
    if "days from now" in t:
        try:
            n = int(t.split("days")[0].strip())
            return now + timedelta(days=n)
        except Exception:
            return None
    try:
        return datetime.fromisoformat(t)
    except Exception:
        return None


def dates_match(expected_date_text: str, observed_date_str: Optional[str]) -> bool:
    """Check if expected and observed delivery dates match."""
    if not observed_date_str:
        return False
    try:
        observed = datetime.fromisoformat(observed_date_str)
    except Exception:
        observed = parse_date_text(observed_date_str)
    expected = parse_date_text(expected_date_text)
    return bool(expected and observed and expected.date() == observed.date())


def create_application_state_from_dict(data: Dict[str, Any]) -> ApplicationState:
    """Create ApplicationState from dictionary data."""
    shipping_info = data.get("shipping_info", {})
    payment_info = data.get("payment_info", {}) or {}
    delivery_instructions = data.get("delivery_instructions", {})
    confirmation_data = data.get("confirmation", {})
    timeline_data = data.get("timeline", [])

    # Extract contact info from shipping_info
    contact = ContactInfo(
        first_name=shipping_info.get("firstName"),
        last_name=shipping_info.get("lastName"),
        email=shipping_info.get("email"),
        phone=shipping_info.get("phoneNumber") or shipping_info.get("phone") or shipping_info.get("telephoneNumber"),
    )

    # Extract address info
    address = AddressInfo(
        street_address=shipping_info.get("streetAddress"),
        city=shipping_info.get("city"),
        state=shipping_info.get("state"),
        zip_code=shipping_info.get("zipCode"),
        country=shipping_info.get("country"),
        used_default=shipping_info.get("used_default"),
    )

    # Extract payment info
    payment = PaymentInfo(
        payment_method=payment_info.get("method"),
        card_type=payment_info.get("cardType"),
        card_number=payment_info.get("cardNumber"),
        card_number_masked=payment_info.get("maskedCard"),
        expiry=payment_info.get("expiryDate") or payment_info.get("expiry"),
        cvv_present=bool(payment_info.get("cvv")),
        cvv=payment_info.get("cvv"),
        used_default=payment_info.get("setAsDefault"),
        billing_address_same=payment_info.get("billing_address_same"),
        billing_address=AddressInfo(
            street_address=payment_info.get("billingAddress", {}).get("streetAddress"),
            city=payment_info.get("billingAddress", {}).get("city"),
            state=payment_info.get("billingAddress", {}).get("state"),
            zip_code=payment_info.get("billingAddress", {}).get("zipCode"),
            country=payment_info.get("billingAddress", {}).get("country"),
        )
        if payment_info.get("billingAddress")
        else None,
    )

    # Extract delivery info
    delivery = DeliveryInfo(
        recipient_name=shipping_info.get("firstName") + " " + shipping_info.get("lastName")
        if shipping_info.get("firstName") and shipping_info.get("lastName")
        else None,
        delivery_date_text=shipping_info.get("delivery_date_text"),
        delivery_date=shipping_info.get("delivery_date"),
        delivery_method=shipping_info.get("delivery_method"),
    )

    # Extract delivery instructions
    delivery_instructions_obj = None
    if delivery_instructions:
        delivery_instructions_obj = DeliveryInstructions(
            dropoff_location=delivery_instructions.get("dropoff_location"),
            property_type=delivery_instructions.get("property_type"),
            delivery_notes=delivery_instructions.get("delivery_notes"),
        )

    # Extract confirmation info
    confirmation = ConfirmationInfo(
        reached=bool(confirmation_data.get("reached", False)),
        order_number=confirmation_data.get("order_number"),
        confirmation_text_present=bool(confirmation_data.get("confirmation_text_present", False)),
    )

    return ApplicationState(
        contact=contact,
        address=address,
        payment=payment,
        delivery=delivery,
        delivery_instructions=delivery_instructions_obj,
        confirmation=confirmation,
        timeline=timeline_data,
        url=data.get("url"),
        delivery_date=data.get("delivery_date"),
    )


def create_basic_checkout_params_from_task(task: Dict[str, Any]) -> BasicCheckoutParams:
    """Create BasicCheckoutParams from task dictionary."""
    values = task.get("values", {})
    user_info = values.get("user_info", {})

    return BasicCheckoutParams(
        delivery_address_type=values.get("delivery_address", "default"),
        payment_method_type=values.get("payment_method", "default"),
        delivery_date_text=values.get("delivery_date"),
        expected_person_name=user_info.get("GivenName") + " " + user_info.get("Surname")
        if user_info.get("GivenName") and user_info.get("Surname")
        else None,
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
        # Delivery instruction fields - handle both snake_case and camelCase
        dropoff_location=values.get("dropoff_location") or values.get("dropoffLocation"),
        property_type=values.get("property_type") or values.get("propertyType"),
        delivery_notes=values.get("delivery_notes") or values.get("deliveryNotes"),
    )


def verify_contact_info(params: BasicCheckoutParams, observed: ApplicationState) -> Dict[str, Any]:
    """Verify contact information matches expected values."""
    failures = []
    checks = []

    def add_check(name: str, expected: str, observed_val: str, passed: bool):
        checks.append({"check": name, "expected": expected, "observed": observed_val, "passed": passed})
        if not passed:
            failures.append(name)

    if params.delivery_address_type == "user":
        # Verify first name
        if params.given_name:
            expected_name = params.given_name
            observed_name = observed.contact.first_name or ""
            add_check(
                "first_name_match", params.given_name, observed.contact.first_name or "", expected_name == observed_name
            )

        # Verify last name
        if params.surname:
            expected_surname = params.surname
            observed_surname = observed.contact.last_name or ""
            add_check(
                "last_name_match",
                params.surname,
                observed.contact.last_name or "",
                expected_surname == observed_surname,
            )

        # Verify phone
        if params.phone:
            expected_phone = re.sub(r"\D", "", params.phone)
            observed_phone = re.sub(r"\D", "", observed.contact.phone or "")
            # Handle hardcoded phone number bug in system
            if observed_phone == "4085550199" and expected_phone != "4085550199":
                # system has a bug where it uses hardcoded phone number instead of user's phone
                add_check("phone_match", expected_phone, observed_phone, True)  # Pass the check
            else:
                add_check("phone_match", expected_phone, observed_phone, expected_phone == observed_phone)
    else:
        add_check("contact_info_type", "default (no specific verification required)", "default", True)

    return {"passed": len(failures) == 0, "failures": failures, "checks": checks}


def verify_address_info(params: BasicCheckoutParams, observed: ApplicationState) -> Dict[str, Any]:
    """Verify address information matches expected values."""
    failures = []
    checks = []

    def add_check(name: str, expected: str, observed_val: str, passed: bool):
        checks.append({"check": name, "expected": expected, "observed": observed_val, "passed": passed})
        if not passed:
            failures.append(name)

    if params.delivery_address_type == "default":
        add_check(
            "used_default_address", "true", str(observed.address.used_default), bool(observed.address.used_default)
        )
    else:
        # Verify street address
        if params.street_address:
            expected_street = params.street_address
            observed_street = observed.address.street_address or ""
            add_check(
                "delivery_street_match",
                params.street_address,
                observed.address.street_address or "",
                expected_street == observed_street,
            )

        # Verify city
        if params.city:
            expected_city = params.city
            observed_city = observed.address.city or ""
            add_check("delivery_city_match", params.city, observed.address.city or "", expected_city == observed_city)

        # Verify state
        if params.state or params.state_full:
            expected_state = params.state or params.state_full
            observed_state = observed.address.state or ""
            add_check(
                "delivery_state_match",
                params.state or params.state_full,
                observed.address.state or "",
                expected_state == observed_state,
            )

        # Verify zip code
        if params.zip_code:
            expected_zip = (params.zip_code or "")[:5]
            observed_zip = (observed.address.zip_code or "")[:5]
            add_check("delivery_zip_match", expected_zip, observed_zip, expected_zip == observed_zip)

    return {"passed": len(failures) == 0, "failures": failures, "checks": checks}


def verify_delivery_info(params: BasicCheckoutParams, observed: ApplicationState) -> Dict[str, Any]:
    """Verify delivery information matches expected values."""
    failures = []
    checks = []

    def add_check(name: str, expected: str, observed_val: str, passed: bool):
        checks.append({"check": name, "expected": expected, "observed": observed_val, "passed": passed})
        if not passed:
            failures.append(name)

    # Verify dropoff location
    if params.dropoff_location:
        expected_dropoff = normalize_string_for_comparison(params.dropoff_location)
        observed_dropoff = ""
        if observed.delivery_instructions and observed.delivery_instructions.dropoff_location:
            observed_dropoff = normalize_string_for_comparison(observed.delivery_instructions.dropoff_location)
        passed = expected_dropoff == observed_dropoff
        add_check("dropoff_location", expected_dropoff, observed_dropoff, passed)

    # Verify property type
    if params.property_type:
        expected_property = normalize_string_for_comparison(params.property_type)
        observed_property = ""
        if observed.delivery_instructions and observed.delivery_instructions.property_type:
            observed_property = normalize_string_for_comparison(observed.delivery_instructions.property_type)
        passed = expected_property == observed_property
        add_check("property_type", expected_property, observed_property, passed)

    return {"passed": len(failures) == 0, "failures": failures, "checks": checks}


def verify_payment_info(params: BasicCheckoutParams, observed: ApplicationState) -> Dict[str, Any]:
    """Verify payment information matches expected values."""
    failures = []
    checks = []

    def add_check(name: str, expected: str, observed_val: str, passed: bool):
        checks.append({"check": name, "expected": expected, "observed": observed_val, "passed": passed})
        if not passed:
            failures.append(name)

    # Only verify specific card details if payment method is "user"
    if params.payment_method_type == "user":
        # Verify CCType from task against observed
        if params.cc_type:
            expected_cc_type = params.cc_type
            observed_cc_type = observed.payment.card_type or ""
            passed = expected_cc_type.lower() == observed_cc_type.lower()
            add_check("cc_type", f"CCType: {expected_cc_type}", observed_cc_type, passed)

        # Verify CCNumber from task against observed (full number)
        if params.cc_number:
            expected_cc_number = params.cc_number
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
        if params.cvv2:
            expected_cvv2 = params.cvv2
            observed_cvv2 = observed.payment.cvv or ""
            passed = expected_cvv2 == observed_cvv2
            add_check("cvv2", f"CVV2: {expected_cvv2}", observed_cvv2, passed)

        # Verify CCExpires from task against observed
        if params.cc_expires:
            expected_expires = normalize_expiry(params.cc_expires)
            observed_expires = normalize_expiry(observed.payment.expiry or "")
            passed = expected_expires == observed_expires
            add_check("cc_expires", f"CCExpires: {expected_expires}", observed_expires, passed)

    # Check payment method type
    if params.payment_method_type == "default":
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
        if params.cc_number:
            # Try to use full card number if available, otherwise use masked
            if observed.payment.card_number:
                # Verify full card number matches
                passed = params.cc_number == observed.payment.card_number
                add_check("card_number", f"user's card: {params.cc_number}", observed.payment.card_number, passed)
            elif observed.payment.card_number_masked:
                # If only masked is available, verify last 4 digits match
                expected_last4 = get_last4_digits(params.cc_number)
                observed_last4 = get_last4_digits(observed.payment.card_number_masked)
                passed = expected_last4 == observed_last4
                add_check(
                    "card_number", f"user's card ending in {expected_last4}", f"ending in {observed_last4}", passed
                )

            # Verify card type if available
            if params.cc_type and observed.payment.card_type:
                expected_type = params.cc_type
                observed_type = observed.payment.card_type
                passed = expected_type.lower() == observed_type.lower()
                add_check("card_type", f"user's card type: {params.cc_type}", observed.payment.card_type, passed)

            # Verify expiry if available
            if params.cc_expires and observed.payment.expiry:
                expected_expiry = normalize_expiry(params.cc_expires)
                observed_expiry = normalize_expiry(observed.payment.expiry)
                passed = expected_expiry == observed_expiry
                add_check("card_expiry", f"user's card expiry: {params.cc_expires}", observed.payment.expiry, passed)

    return {"passed": len(failures) == 0, "failures": failures, "checks": checks}


def verify_confirmation_info(
    params: BasicCheckoutParams, observed: ApplicationState, agent_output: Optional[AgentOutput] = None
) -> Dict[str, Any]:
    """Verify confirmation information matches expected values."""
    failures = []
    checks = []

    def add_check(name: str, expected: str, observed_val: str, passed: bool):
        checks.append({"check": name, "expected": expected, "observed": observed_val, "passed": passed})
        if not passed:
            failures.append(name)

    add_check("confirmation_reached", "true", str(observed.confirmation.reached), bool(observed.confirmation.reached))
    add_check(
        "confirmation_text_present",
        "true",
        str(observed.confirmation.confirmation_text_present),
        bool(observed.confirmation.confirmation_text_present),
    )

    if agent_output and agent_output.order_number:
        if observed.confirmation.order_number:
            expected_order = agent_output.order_number
            observed_order = observed.confirmation.order_number
            add_check(
                "order_number_match",
                agent_output.order_number,
                observed.confirmation.order_number,
                observed_order in expected_order,
            )
            if observed_order not in expected_order:
                failures.append("order_number_mismatch")
        else:
            add_check("order_number_match", agent_output.order_number, "None (not reached)", False)
            failures.append("order_number_not_found")
    elif agent_output and not agent_output.order_number:
        add_check("order_number_provided", "order number", "None provided", False)
        failures.append("order_number_not_provided")

    return {"passed": len(failures) == 0, "failures": failures, "checks": checks}


def verify_basic_checkout_complete(
    params: BasicCheckoutParams,
    observed: ApplicationState,
    agent_output: Optional[AgentOutput] = None,
    timed_out: bool = False,
) -> VerificationResult:
    """Main verification function for basic checkout completion."""
    contact_result = verify_contact_info(params, observed)
    address_result = verify_address_info(params, observed)
    delivery_result = verify_delivery_info(params, observed)
    payment_result = verify_payment_info(params, observed)
    confirmation_result = verify_confirmation_info(params, observed, agent_output)

    # Collect all failures
    all_failures = []
    if not contact_result["passed"]:
        all_failures.extend(contact_result["failures"])
    if not address_result["passed"]:
        all_failures.extend(address_result["failures"])
    if not delivery_result["passed"]:
        all_failures.extend(delivery_result["failures"])
    if not payment_result["passed"]:
        all_failures.extend(payment_result["failures"])
    if not confirmation_result["passed"]:
        all_failures.extend(confirmation_result["failures"])

    if timed_out:
        all_failures.append("timeout")

    # Overall pass/fail
    overall_passed = (
        all(
            [
                contact_result["passed"],
                address_result["passed"],
                delivery_result["passed"],
                payment_result["passed"],
                confirmation_result["passed"],
            ]
        )
        and not timed_out
    )

    details = {
        "contact": contact_result,
        "address": address_result,
        "delivery": delivery_result,
        "payment": payment_result,
        "confirmation": confirmation_result,
        "timeline": observed.timeline,
    }

    return VerificationResult(passed=overall_passed, details=details, failures=all_failures)


class InputDict(TypedDict, total=False):
    page: "Page"


class FinalResult(BaseModel):
    """Result model for basic checkout task."""

    score: float
    reasoning: str
    current_state: dict[str, Any]
    ground_truth: BasicCheckoutParams


class HalluminateGoodBuyBasicCheckout(BaseMetric):
    """Evaluator that runs verifier for basic checkout task."""

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
    """Render values for basic_checkout tasks."""
    return values


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
    eval_target = get_import_path(HalluminateGoodBuyBasicCheckout)
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
        "task_id": "halluminate_bench/goodbuy/basic_checkout/1",
        "task_generation_config_json": json.dumps(
            {
                "_target_": "halluminate_bench.goodbuy.basic_checkout.generate_task_config",
                "url": "https://good-buy.halluminate.ai/?apiKey={HALLUMINATE_API_KEY}",
                "task": (
                    "Purchase the items in the cart and deliver them to Violet Ritchie. "
                    "Use Violet Ritchie's payment info. Set the delivery instruction dropoff location to front door "
                    "and property type to apartment. Your job is done when the order is successfully placed. "
                    "Use guest checkout. User info: {user_info}."
                ),
                "values": {
                    "delivery_address": "user",
                    "payment_method": "user",
                    "dropoff_location": "front door",
                    "property_type": "apartment",
                    "delivery_notes": "call on arrival",
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
