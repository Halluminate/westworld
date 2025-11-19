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
class DeliveryInstructionInfo:
    """Delivery instruction information for the checkout."""

    package_location: Optional[str] = None  # "front-door", "outside-garage", etc.
    security_code: Optional[str] = None  # "1234", "5678", "91011"
    call_box: Optional[str] = None  # Call box information
    key_required: Optional[bool] = None  # Whether key is required
    additional_instructions: Optional[str] = None  # Additional delivery notes


@dataclass
class CartInfo:
    """Cart information for the checkout."""

    items: Optional[List[Dict[str, Any]]] = None
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
    delivery_instructions: DeliveryInstructionInfo
    cart: CartInfo
    confirmation: ConfirmationInfo
    timeline: List[Dict[str, Any]]
    current_url: Optional[str] = None
    current_page: Optional[str] = None  # "cart", "checkout", "confirmation", etc.
    form_states: Optional[Dict[str, bool]] = None


@dataclass
class BasicDeliveryInstructionParams:
    """Expected delivery instruction parameters from the basic delivery instruction task."""

    # Extracted from task description
    delivery_recipient: Optional[str] = None  # Jennifer Morrison, Sharon Samples, etc.
    delivery_location: Optional[str] = None  # "front door", "outside garage"
    security_code: Optional[str] = None  # "1234", "5678", "91011"

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


def normalize_string_for_comparison(text: str) -> str:
    """Normalize string for case-insensitive comparison."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip().lower()


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


def extract_delivery_location_from_task(task_description: str) -> Optional[str]:
    """Extract delivery location from task description.

    Args:
        task_description: The task description text

    Returns:
        Delivery location if found, None otherwise
    """
    task_lower = task_description.lower()

    if "front door" in task_lower:
        return "front-door"  # Match the format used in the state
    elif "outside garage" in task_lower:
        return "outside-garage"  # Match the format used in the state

    return None


def extract_security_code_from_task(task_description: str) -> Optional[str]:
    """Extract security code from task description.

    Args:
        task_description: The task description text

    Returns:
        Security code if found, None otherwise
    """
    # Look for security code patterns in the task description
    # Pattern: "security code to get in is XXXX"
    match = re.search(r"security code.*?is\s+(\d+)", task_description, re.IGNORECASE)
    if match:
        return match.group(1)

    return None


def parse_security_code_value(value: Optional[str]) -> Optional[str]:
    """
    Normalize the observed security_code field which may contain either
    plain digits (e.g., '1234') or free-text (e.g., 'Security code: 1234').
    Returns the first 3–10 digit sequence found, or None if not found.
    """
    if value is None:
        return None
    # Prefer the explicit phrase, but also handle plain digits only
    m = re.search(r"security\s*code.*?(\d{3,10})", str(value), re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"\b(\d{3,10})\b", str(value))
    return m.group(1) if m else None


def verify_delivery_recipient(params: BasicDeliveryInstructionParams, observed: ApplicationState) -> Dict[str, Any]:
    """Verify delivery recipient matches expected name."""
    failures = []
    checks = []

    def add_check(name: str, expected: str, observed_val: str, passed: bool):
        checks.append({"check": name, "expected": expected, "observed": observed_val, "passed": passed})
        if not passed:
            failures.append(name)

    if params.delivery_recipient:
        expected_recipient = normalize_string_for_comparison(params.delivery_recipient)
        observed_recipient = normalize_string_for_comparison(observed.address.full_name or "")
        passed = expected_recipient == observed_recipient
        add_check("delivery_recipient_match", params.delivery_recipient, observed.address.full_name or "", passed)
    else:
        add_check("delivery_recipient_extraction", "recipient name from task", "none extracted", False)

    return {"passed": len(failures) == 0, "failures": failures, "checks": checks}


def verify_delivery_instructions(params: BasicDeliveryInstructionParams, observed: ApplicationState) -> Dict[str, Any]:
    """Verify delivery instructions were properly set."""
    failures = []
    checks = []

    def add_check(name: str, expected: str, observed_val: str, passed: bool):
        checks.append({"check": name, "expected": expected, "observed": observed_val, "passed": passed})
        if not passed:
            failures.append(name)

    # Check delivery location (front door, outside garage, etc.)
    if params.delivery_location:
        if observed.delivery_instructions.package_location:
            observed_location = normalize_string_for_comparison(observed.delivery_instructions.package_location)
            expected_location = normalize_string_for_comparison(params.delivery_location)
            passed = expected_location == observed_location
            add_check(
                "delivery_location_match",
                params.delivery_location,
                observed.delivery_instructions.package_location,
                passed,
            )
        else:
            add_check("delivery_instructions_present", params.delivery_location, "None", False)
    else:
        add_check("delivery_location_extraction", "delivery location from task", "none extracted", False)

    # Check security code
    raw_code = observed.delivery_instructions.security_code
    observed_code = parse_security_code_value(raw_code)
    if params.security_code:
        if observed_code:
            passed = params.security_code == observed_code
            add_check("security_code_match", params.security_code, observed_code, passed)
        else:
            add_check("security_code_present", params.security_code, "None", False)
    else:
        add_check("security_code_extraction", "security code from task", "none extracted", False)

    return {"passed": len(failures) == 0, "failures": failures, "checks": checks}


def verify_default_payment_and_delivery(
    params: BasicDeliveryInstructionParams, observed: ApplicationState
) -> Dict[str, Any]:
    """Verify default payment method and delivery date were used."""
    failures = []
    checks = []

    def add_check(name: str, expected: str, observed_val: str, passed: bool):
        checks.append({"check": name, "expected": expected, "observed": observed_val, "passed": passed})
        if not passed:
            failures.append(name)

    # Check if default payment method was used
    payment_default_used = observed.payment.used_default
    if payment_default_used is not None:
        add_check("default_payment_used", "true", str(payment_default_used), payment_default_used)
    else:
        # If used_default is not tracked, check if payment method type is present
        payment_method_present = bool(observed.payment.payment_method_type)
        add_check("payment_method_present", "true", str(payment_method_present), payment_method_present)

    # For delivery date, the task specifies using the default delivery date
    # We'll be lenient since the task doesn't specify a particular date
    if observed.shipping.delivery_date:
        add_check("delivery_date_selected", "any valid date", observed.shipping.delivery_date, True)
    else:
        add_check("delivery_date_selected", "any valid date", "None", False)

    return {"passed": len(failures) == 0, "failures": failures, "checks": checks}


def verify_confirmation(
    params: BasicDeliveryInstructionParams, observed: ApplicationState, agent_output: Optional[AgentOutput]
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

    delivery_instructions = DeliveryInstructionInfo(
        package_location=data.get("delivery_instructions", {}).get("package_location"),
        security_code=data.get("delivery_instructions", {}).get("security_code"),
        call_box=data.get("delivery_instructions", {}).get("call_box"),
        key_required=data.get("delivery_instructions", {}).get("key_required"),
        additional_instructions=data.get("delivery_instructions", {}).get("additional_instructions"),
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
        delivery_instructions=delivery_instructions,
        cart=cart,
        confirmation=confirmation,
        timeline=data.get("timeline", []),
        current_url=data.get("current_url"),
        current_page=data.get("current_page"),
        form_states=data.get("form_states"),
    )


def create_basic_delivery_instruction_params_from_task(task_data: Dict[str, Any]) -> BasicDeliveryInstructionParams:
    """Create BasicDeliveryInstructionParams from task data."""
    task_description = task_data.get("task", "")
    user_info = task_data.get("values", {}).get("user_info", {})
    expected_values = task_data.get("expected_values", {})

    # Use explicit expected values if available, otherwise fall back to text parsing
    delivery_recipient = expected_values.get("delivery_recipient") or extract_recipient_from_task(task_description)
    delivery_location = expected_values.get("delivery_location") or extract_delivery_location_from_task(
        task_description
    )
    security_code = expected_values.get("security_code") or extract_security_code_from_task(task_description)

    return BasicDeliveryInstructionParams(
        delivery_recipient=delivery_recipient,
        delivery_location=delivery_location,
        security_code=security_code,
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


def verify_basic_delivery_instruction_complete(
    params: BasicDeliveryInstructionParams, observed: ApplicationState, agent_output: Optional[AgentOutput] = None
) -> VerificationResult:
    """Verify that the basic delivery instruction task was completed successfully."""
    all_failures = []
    details = {}

    # Verify delivery recipient
    recipient_result = verify_delivery_recipient(params, observed)
    details["delivery_recipient"] = recipient_result
    all_failures.extend(recipient_result["failures"])

    # Verify delivery instructions and security code
    instructions_result = verify_delivery_instructions(params, observed)
    details["delivery_instructions"] = instructions_result
    all_failures.extend(instructions_result["failures"])

    # Verify default payment and delivery were used
    default_result = verify_default_payment_and_delivery(params, observed)
    details["default_payment_and_delivery"] = default_result
    all_failures.extend(default_result["failures"])

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
    """Result model for basic delivery instruction task."""

    score: float
    reasoning: str
    current_state: dict[str, Any]
    ground_truth: Any


class HalluminateAzoraBasicDeliveryInstruction(BaseMetric):
    """Evaluator that runs verifier for Azora basic delivery instruction task."""

    def __init__(
        self, task: str, values: dict[str, Any], expected_values: dict[str, Any] | None, user_metadata: UserMetadata
    ) -> None:
        super().__init__()
        self.task = task
        self.values = values
        self.expected_values = expected_values or {}
        self.user_metadata = user_metadata
        # Create ground truth from task data
        # (verifier expects {"task": task, "values": values, "expected_values": expected_values})
        self._ground_truth = create_basic_delivery_instruction_params_from_task(
            {
                "task": task,
                "values": values,
                "expected_values": self.expected_values,
            }
        )
        self._page: "Page" | None = None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(task={self.task}, values={self.values}, "
            f"expected_values={self.expected_values}, user_metadata={self.user_metadata})"
        )

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
        result = verify_basic_delivery_instruction_complete(self._ground_truth, state_obj, agent_output=None)

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
                            detail = f"  {check_name}: expected '{expected}', observed '{observed}'"
                            failure_details.append(detail)

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


def _render_values(values: dict[str, Any], timestamp: int) -> dict[str, Any]:
    """Render values for basic_delivery_instruction tasks.

    For basic_delivery_instruction, we don't have date formulas (uses default),
    but we keep this function for consistency with other evaluators.
    """
    return dict(values)


def _render_task(task: str, values: dict[str, Any], expected_values: dict[str, Any]) -> str:
    """Replace {delivery_recipient}, {delivery_location}, {security_code}, {user_info} placeholders.

    Example:
        Input task: "deliver them to {delivery_recipient}. Set the delivery instructions
                    so the package is left at {delivery_location}..."
        Input values: {"delivery_recipient": "Jennifer Morrison", "delivery_location": "front-door", ...}
        Output: "deliver them to Jennifer Morrison. Set the delivery instructions
                so the package is left at the front door..."
    """
    result = task

    # Replace {delivery_recipient}
    if "{delivery_recipient}" in result:
        delivery_recipient = expected_values.get("delivery_recipient", "") or values.get("delivery_recipient", "")
        result = result.replace("{delivery_recipient}", delivery_recipient)

    # Replace {delivery_location}
    if "{delivery_location}" in result:
        delivery_location = expected_values.get("delivery_location", "") or values.get("delivery_location", "")
        # Convert location code to readable text
        location_map = {
            "front-door": "left at the front door",
            "outside-garage": "left outside the garage",
            "back-door": "left at the back door",
        }
        location_text = location_map.get(delivery_location, f"left at {delivery_location}")
        result = result.replace("{delivery_location}", location_text)

    # Replace {security_code}
    if "{security_code}" in result:
        security_code = expected_values.get("security_code", "") or values.get("security_code", "")
        result = result.replace("{security_code}", security_code)

    # Replace {user_info} with string representation
    if "{user_info}" in result:
        user_info = values.get("user_info", {})
        result = result.replace("{user_info}", str(user_info))

    return result


def generate_task_config(
    url: str,
    task: str,
    values: dict[str, Any],
    expected_values: dict[str, Any] | None,
    location: str,
    timezone: str,
    timestamp: int | None = None,
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
        values: Task values (user_info)
        expected_values: Expected values (delivery_recipient, delivery_location, security_code)
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

    # Render the values (no date formulas for delivery instruction, but keep for consistency)
    values = _render_values(values, timestamp)

    # Render the task (replace placeholders with formatted values)
    expected_values = expected_values or {}
    task = _render_task(task, values, expected_values)

    # Create the eval config
    eval_target = get_import_path(HalluminateAzoraBasicDeliveryInstruction)
    eval_config = {
        "_target_": eval_target,
        "task": task,
        "values": values,
        "expected_values": expected_values,
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
        "task_id": "halluminate_bench/azora/basic_delivery_instruction/1",
        "task_generation_config_json": json.dumps(
            {
                "_target_": "halluminate_bench.azora.basic_delivery_instruction.generate_task_config",
                "url": "https://azora.halluminate.ai/?apiKey={HALLUMINATE_API_KEY}",
                "task": (
                    "Purchase these Airpods and deliver them to {delivery_recipient}. "
                    "Set the delivery instructions so the package is left at {delivery_location}. "
                    "The security code to get in is {security_code}. "
                    "Use the default payment method and delivery date. "
                    "Your job is done when the order is successfully placed. User info: {user_info}"
                ),
                "values": {
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
                "expected_values": {
                    "delivery_recipient": "Jennifer Morrison",
                    "delivery_location": "front-door",
                    "security_code": "1234",
                },
                "location": "San Francisco, CA, United States",
                "timezone": "America/Los_Angeles",
                "timestamp": None,
            }
        ),
        "env": "sim",
        "domain": "azora",
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
