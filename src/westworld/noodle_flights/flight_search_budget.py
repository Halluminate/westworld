import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Dict, List, Optional, TypedDict
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import requests
from pydantic import BaseModel

from westworld.base import (
    BaseMetric,
    BaseTaskConfig,
    DatasetItem,
    UserMetadata,
    async_retry_with_exponential_backoff,
    get_import_path,
)


if TYPE_CHECKING:
    from playwright.async_api import Page


# =========================
# Data models
# =========================


@dataclass
class ApplicationState:
    """Current state of the flight search application (from window.__testingAPI.getState())."""

    trip_type: str
    from_location: str
    to_location: str
    departure_date: Optional[str]
    travel_class: str
    is_search_ready: bool
    has_multiple_from_locations: bool
    has_multiple_to_locations: bool


@dataclass
class BudgetSearchParams:
    """Expected flight search params from the task."""

    origin: Optional[str] = None
    destination: Optional[str] = None
    trip_type: Optional[str] = None
    date: Optional[str] = None
    travel_class: Optional[str] = None
    budget: Optional[float] = None

    # Optional fields for more complex tasks
    origin_airport: Optional[str] = None
    origin_city: Optional[str] = None
    origin_has_multiple_airports: Optional[bool] = None
    destination_airport: Optional[str] = None
    destination_city: Optional[str] = None
    destination_has_multiple_airports: Optional[bool] = None


@dataclass
class VerificationResult:
    """Result of the verification process."""

    passed: bool
    details: Dict[str, Any]
    message: str = ""


# =========================
# API helpers
# =========================


def query_noodle_flights_simulator(
    origin: str, destination: str, date: str, api_base_url: Optional[str] = None
) -> Dict[str, Any]:
    """
    Query the Noodle Flights simulator to get available flights.

    Args:
        origin: Origin airport code (e.g., "NNG")
        destination: Destination airport code (e.g., "YYZ")
        date: Departure date in YYYY-MM-DD format

    Returns:
        Dictionary containing flight data from the simulator
    """
    provided_base = api_base_url or os.getenv("FLIGHTS_BASE_URL") or "https://flights.halluminate.ai"

    if provided_base.rstrip("/").endswith("/api/flights"):
        flights_endpoint = provided_base.rstrip("/")
    else:
        flights_endpoint = provided_base.rstrip("/") + "/api/flights"

    params = {
        "origin": origin,
        "destination": destination,
        "departureDate": date,
        "tripType": "one-way",
        "travelClass": "economy",
    }

    try:
        response = requests.get(f"{flights_endpoint}?{urlencode(params)}", timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to query flights API at {flights_endpoint}: {e}")


def check_budget_constraint(flights: list, budget: int) -> bool:
    """
    Check if any flights are available under the budget constraint.

    Args:
        flights: List of flight objects from the simulator
        budget: Maximum budget in USD

    Returns:
        True if at least one flight is under budget, False otherwise
    """

    if not flights:
        return False

    for flight in flights:
        if flight.get("priceUsd", float("inf")) < budget:
            return True

    return False


def get_runtime_ground_truth(task_data: Dict[str, Any], api_base_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Get ground truth by querying the simulator at runtime.

    Args:
        task_data: Task data containing origin, destination, date, and budget

    Returns:
        Ground truth information from the simulator
    """
    values = task_data.get("values", {})
    origin = values.get("origin")
    destination = values.get("destination")
    date = values.get("date")
    budget = values.get("budget")

    if not all([origin, destination, date, budget]):
        raise ValueError("Task missing required fields: origin, destination, date, or budget")

    try:
        # Query the simulator
        flight_data = query_noodle_flights_simulator(origin, destination, date, api_base_url)

        # Check if there are flights under budget
        flights = flight_data.get("departingFlights", [])
        has_options_under_budget = check_budget_constraint(flights, budget)

        return {
            "hasOptionsUnderBudget": has_options_under_budget,
            "totalFlights": len(flights),
            "cheapestFlight": min([f.get("priceUsd", float("inf")) for f in flights]) if flights else None,
            "budget": budget,
        }
    except Exception as e:
        return {
            "hasOptionsUnderBudget": False,
            "totalFlights": 0,
            "cheapestFlight": None,
            "budget": budget,
            "error": str(e),
        }


# =========================
# Verification helpers
# =========================


def is_specified(value: Any) -> bool:
    """Check if a value was specified (not None or empty string)."""
    if value is None:
        return False
    if isinstance(value, str) and value.strip() == "":
        return False
    return True


def normalize_string_for_comparison(text: Optional[str]) -> str:
    """Normalize string for case-insensitive comparison."""
    return (text or "").strip().lower()


def dates_match(date1: Optional[str], date2: Optional[str]) -> bool:
    """Compare two dates in ISO format (YYYY-MM-DD)."""
    try:
        d1 = datetime.fromisoformat(str(date1).split("T")[0])
        d2 = datetime.fromisoformat(str(date2).split("T")[0])
        return d1.date() == d2.date()
    except (ValueError, AttributeError):
        return False


def bidirectional_substring_match(expected: Optional[str], actual: Optional[str]) -> bool:
    e = normalize_string_for_comparison(expected)
    a = normalize_string_for_comparison(actual)
    return (e in a) or (a in e)


def location_matches_with_sophisticated_fallback(
    expected_code: Optional[str],
    expected_airport: Optional[str],
    expected_city: Optional[str],
    has_multiple_airports: Optional[bool],
    actual: Optional[str],
) -> bool:
    """
    Check if location matches with sophisticated fallback logic.

    Logic:
    1. Always try to match airport code first
    2. Always try to match airport name
    3. For cities with multiple airports: only accept code/airport name matches
    4. For cities with single airport: also accept city name matches
    """

    if expected_code and bidirectional_substring_match(expected_code, actual):
        return True
    if expected_airport and bidirectional_substring_match(expected_airport, actual):
        return True
    if has_multiple_airports:
        return False
    if expected_city and bidirectional_substring_match(expected_city, actual):
        return True
    return False


def verify_agent_response(
    agent_response: str, task_id: str, task_data: Dict[str, Any], *, base_url: Optional[str] = None
) -> VerificationResult:
    """
    Verify a budget flight search task by querying the simulator at runtime.

    Args:
        agent_response: The agent's response (should contain "yes" or "no")
        task_id: The task ID
        task_data: Task data containing origin, destination, date, and budget

    Returns:
        Verification result
    """
    ground_truth = get_runtime_ground_truth(task_data, base_url)

    # Define exact expected phrases (case-insensitive matching)
    EXACT_YES_PHRASE = "yes, there is at least one option"
    EXACT_NO_PHRASE = "no options found"

    # Normalize response
    normalized_response = re.sub(r"\s+", " ", agent_response.lower().strip())

    # Check for exact phrases
    yes_pattern = r"\b" + re.escape(EXACT_YES_PHRASE) + r"(?:\s*[\.!?]?\s*|$)"
    no_pattern = r"\b" + re.escape(EXACT_NO_PHRASE) + r"(?:\s*[\.!?]?\s*|$)"

    has_yes = bool(re.search(yes_pattern, normalized_response))
    has_no = bool(re.search(no_pattern, normalized_response))

    is_unambiguous = (has_yes or has_no) and not (has_yes and has_no)
    response_answer = has_yes

    is_correct = is_unambiguous and response_answer == ground_truth["hasOptionsUnderBudget"]

    expected_answer_phrase = (
        "Yes, there is at least one option" if ground_truth["hasOptionsUnderBudget"] else "No options found"
    )

    return VerificationResult(
        passed=is_correct,
        details={
            "expectedAnswer": expected_answer_phrase,
            "providedAnswer": "invalid" if not is_unambiguous else ("yes" if response_answer else "no"),
            "foundYes": has_yes,
            "foundNo": has_no,
            "isUnambiguous": is_unambiguous,
            "expectedYesPhrase": EXACT_YES_PHRASE,
            "expectedNoPhrase": EXACT_NO_PHRASE,
            "runtimeVerification": True,
            "totalFlights": ground_truth.get("totalFlights", 0),
            "cheapestFlight": ground_truth.get("cheapestFlight"),
            "budget": ground_truth.get("budget"),
            "error": ground_truth.get("error"),
        },
        message=(
            "Correct answer!"
            if is_correct
            else f"Invalid response. Response must contain exactly '{EXACT_YES_PHRASE}' or '{EXACT_NO_PHRASE}'."
            if not is_unambiguous
            else "Incorrect answer."
        ),
    )


def verify_flight_search_budget_complete(
    params: BudgetSearchParams,
    state: ApplicationState,
    agent_response: Optional[str] = None,
    *,
    base_url: Optional[str] = None,
) -> VerificationResult:
    """
    Verify that the application state matches the expected flight search parameters,
    and validate the agent's yes/no response against simulator data.

    Args:
        params: Expected parameters from the task
        state: Current application state
        agent_response: agent response ("yes" / "no") to validate
        base_url: Optional base URL for the flights API

    Returns:
        VerificationResult with combined UI state and agent response validation results
    """
    errors: List[str] = []

    # Check trip type if specified
    if is_specified(params.trip_type):
        if normalize_string_for_comparison(state.trip_type) != normalize_string_for_comparison(params.trip_type):
            errors.append(f'Trip type mismatch. Expected "{params.trip_type}", found "{state.trip_type}"')

    # Check origin if specified
    if is_specified(params.origin):
        if not location_matches_with_sophisticated_fallback(
            params.origin,
            params.origin_airport,
            params.origin_city,
            params.origin_has_multiple_airports,
            state.from_location,
        ):
            expected_display = f'"{params.origin}"'
            if params.origin_airport:
                expected_display += f' or "{params.origin_airport}"'
            if params.origin_city and not params.origin_has_multiple_airports:
                expected_display += f' or "{params.origin_city}"'
            errors.append(f'Origin location mismatch. Expected {expected_display}, found "{state.from_location}"')

    # Check destination if specified
    if is_specified(params.destination):
        if not location_matches_with_sophisticated_fallback(
            params.destination,
            params.destination_airport,
            params.destination_city,
            params.destination_has_multiple_airports,
            state.to_location,
        ):
            expected_display = f'"{params.destination}"'
            if params.destination_airport:
                expected_display += f' or "{params.destination_airport}"'
            if params.destination_city and not params.destination_has_multiple_airports:
                expected_display += f' or "{params.destination_city}"'
            errors.append(f'Destination mismatch. Expected {expected_display}, found "{state.to_location}"')

    # Check departure date if specified
    if is_specified(params.date):
        if not state.departure_date:
            errors.append("Departure date not set")
        elif not dates_match(params.date, state.departure_date):
            errors.append(f"Departure date mismatch. Expected {params.date}, found {state.departure_date}")

    # Check travel class if specified
    if is_specified(params.travel_class):
        if normalize_string_for_comparison(state.travel_class) != normalize_string_for_comparison(params.travel_class):
            errors.append(f'Travel class mismatch. Expected "{params.travel_class}", found "{state.travel_class}"')

    # Check if the form is ready
    if not state.is_search_ready:
        errors.append("Search form is not complete - missing required fields")

    # BUDGET VALIDATION: Query the simulator to get real flight data
    budget_info = {}
    if params.budget is not None and params.origin and params.destination and params.date:
        try:
            task_data = {
                "values": {
                    "origin": params.origin,
                    "destination": params.destination,
                    "date": params.date,
                    "budget": params.budget,
                }
            }

            ground_truth = get_runtime_ground_truth(task_data, base_url)

            budget_info = {
                "runtime_verification": True,
                "simulator_total_flights": ground_truth.get("totalFlights", 0),
                "simulator_cheapest_flight": ground_truth.get("cheapestFlight"),
                "simulator_has_options_under_budget": ground_truth.get("hasOptionsUnderBudget"),
            }

            if ground_truth.get("error"):
                budget_info["simulator_error"] = ground_truth.get("error")

        except Exception as e:
            budget_info = {"runtime_verification": False, "simulator_error": str(e)}
    else:
        budget_info = {
            "runtime_verification": False,
            "simulator_error": "Missing required fields for budget validation",
        }

    # AGENT RESPONSE VALIDATION
    agent_validation: Dict[str, Any] = {}
    agent_passed = True

    if len(errors) == 0 and params.budget is not None:
        if agent_response is not None:
            try:
                if budget_info.get("runtime_verification"):
                    agent_result = verify_agent_response(
                        agent_response,
                        "task",
                        {
                            "values": {
                                "origin": params.origin,
                                "destination": params.destination,
                                "date": params.date,
                                "budget": params.budget,
                            }
                        },
                        base_url=base_url,
                    )

                    agent_validation = {
                        "agent_response": agent_response,
                        "expected_answer": agent_result.details.get("expectedAnswer"),
                        "is_correct": agent_result.passed,
                    }
                    agent_passed = agent_result.passed
                else:
                    agent_validation = {
                        "agent_response": agent_response,
                        "error": "Budget validation failed, cannot verify agent response",
                        "is_correct": False,
                    }
                    agent_passed = False
            except Exception as e:
                agent_validation = {"error": f"Agent validation failed: {str(e)}", "is_correct": False}
                agent_passed = False
        else:
            agent_validation = {"error": "No agent response provided", "is_correct": False}
            agent_passed = False

    ui_passed = len(errors) == 0
    overall_passed = ui_passed and agent_passed

    actual_summary: Dict[str, Any] = {
        "trip_type": state.trip_type,
        "from_location": state.from_location,
        "to_location": state.to_location,
        "departure_date": state.departure_date,
        "travel_class": state.travel_class,
        "is_search_ready": state.is_search_ready,
        "agent_validation": agent_validation,
    }

    expected_summary = {
        "origin": params.origin,
        "destination": params.destination,
        "date": params.date,
        "trip_type": "one-way",
        "travel_class": "economy",
        "budget": params.budget,
    }

    if overall_passed:
        message = "Passed all checks"
    else:
        if not ui_passed:
            message = f"Failed (UI): {len(errors)} errors"
        elif not agent_passed:
            if agent_response is None:
                message = "Failed (agent): No agent response provided"
            else:
                message = f"Failed (agent): {agent_validation.get('error', 'Incorrect answer')}"
        else:
            message = "Failed (unexpected)"

    return VerificationResult(
        passed=overall_passed,
        details={"errors": errors, "expected": expected_summary, "actual": actual_summary},
        message=message,
    )


def create_application_state_from_dict(data: Dict[str, Any]) -> ApplicationState:
    """Map window.__testingAPI.getState() to ApplicationState."""
    trip_type = data.get("tripType", "") or data.get("trip_type", "")
    from_loc = data.get("fromLocation", "") or data.get("from_location", "")
    to_loc = data.get("toLocation", "") or data.get("to_location", "")
    dep_date = data.get("departureDate") or data.get("departure_date")
    tclass = data.get("travelClass", "") or data.get("travel_class", "")
    ready = bool(data.get("isSearchReady", False))

    return ApplicationState(
        trip_type=trip_type,
        from_location=from_loc,
        to_location=to_loc,
        departure_date=dep_date,
        travel_class=tclass,
        is_search_ready=ready,
        has_multiple_from_locations=bool(data.get("hasMultipleFromLocations", False)),
        has_multiple_to_locations=bool(data.get("hasMultipleToLocations", False)),
    )


def create_flight_search_budget_params_from_dict(data: Dict[str, Any]) -> BudgetSearchParams:
    """Create a BudgetSearchParams object from a dictionary."""
    return BudgetSearchParams(
        origin=data.get("origin"),
        destination=data.get("destination"),
        date=data.get("date") or data.get("departureDate"),
        budget=float(data["budget"]) if "budget" in data and data["budget"] is not None else None,
        trip_type=data.get("tripType"),
        travel_class=data.get("class") or data.get("travelClass"),
        origin_airport=data.get("originAirport"),
        origin_city=data.get("originCity"),
        origin_has_multiple_airports=data.get("originHasMultipleAirports"),
        destination_airport=data.get("destinationAirport"),
        destination_city=data.get("destinationCity"),
        destination_has_multiple_airports=data.get("destinationHasMultipleAirports"),
    )


# =========================
# Evaluator
# =========================


class InputDict(TypedDict, total=False):
    page: "Page"
    answer_message: str


class FinalResult(BaseModel):
    """Result model for Noodle Flights flight search budget task."""

    score: float
    reasoning: str
    current_state: dict[str, Any]
    ground_truth: BudgetSearchParams


class HalluminateNoodleFlightsFlightSearchBudget(BaseMetric):
    """Evaluator that runs verifier for Noodle Flights flight search budget task."""

    def __init__(self, task: str, values: dict[str, Any], user_metadata: UserMetadata) -> None:
        super().__init__()
        self.task = task
        self.values = values
        self.user_metadata = user_metadata

        self._ground_truth = create_flight_search_budget_params_from_dict(values)
        self._page: "Page" | None = None
        self._answer_message: str | None = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(task={self.task}, values={self.values}, user_metadata={self.user_metadata})"

    async def reset(self) -> None:
        """Reset the evaluator state."""
        self._page = None
        self._answer_message = None

    async def update(self, /, **kwargs):
        inputs: InputDict = kwargs
        self._page = inputs.get("page")
        self._answer_message = inputs.get("answer_message")

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
        result = verify_flight_search_budget_complete(
            self._ground_truth, state_obj, agent_response=self._answer_message
        )

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
                reasoning="Failed (verification): " + "; ".join(result.details["errors"]),
                current_state=current_state,
                ground_truth=self._ground_truth,
            )

    @async_retry_with_exponential_backoff()
    async def _get_current_state(self) -> dict[str, Any]:
        return await self._page.evaluate("() => window.__testingAPI.getState()")


# =========================
# Task generation helpers
# =========================


def _evaluate_date_formula(formula_str: str, timezone: str, timestamp: int) -> datetime:
    """Safely evaluate a date formula string and return datetime object.

    Example:
        Input:  formula_str="now() + timedelta(days=45)", timezone="America/Los_Angeles", timestamp=1715702400
        Output: datetime object

        Input:  formula_str="now() + timedelta(days=45)", timezone="Asia/Tokyo", timestamp=1715702400
        Output: datetime object
    """

    def now():
        return datetime.fromtimestamp(timestamp, tz=ZoneInfo(timezone))

    safe_globals = {"timedelta": timedelta, "now": now}
    result = eval(formula_str, safe_globals)
    return result


def _format_date_for_task(date_obj: datetime) -> str:
    """Convert datetime to 'Month Day, Year' format for task statement.

    Example:
        Input:  datetime(2025, 12, 4)
        Output: "December 4, 2025"
    """
    return date_obj.strftime("%B %d, %Y")


def _render_values(values: dict[str, Any], timezone: str, timestamp: int) -> dict[str, Any]:
    """Render date formula for flight_search_budget tasks.

    Converts formula string to YYYY-MM-DD format string.
    All other fields are left unchanged.
    """
    evaluated = dict(values)

    # Evaluate date if it's a formula
    date_val = evaluated.get("date")
    if isinstance(date_val, str) and ("now()" in date_val or "timedelta" in date_val):
        date_dt = _evaluate_date_formula(date_val, timezone, timestamp)
        evaluated["date"] = date_dt.strftime("%Y-%m-%d")

    return evaluated


def _render_task(task: str, values: dict[str, Any], timestamp: int) -> str:
    """Replace {date} placeholder with formatted date.

    Example:
        Input task: "Are there any flights on {date}..."
        Input values: {"date": "2025-12-04"}
        Output: "Are there any flights on December 4, 2025..."
    """
    result = task

    # Replace {date} placeholder
    if "{date}" in result:
        date_val = values.get("date")
        if isinstance(date_val, str) and len(date_val) == 10:  # YYYY-MM-DD format
            try:
                date_dt = datetime.strptime(date_val, "%Y-%m-%d")
                result = result.replace("{date}", _format_date_for_task(date_dt))
            except ValueError:
                # If parsing fails, leave placeholder as-is
                pass

    return result


def generate_task_config(
    url: str, task: str, values: dict[str, Any], location: str, timezone: str, timestamp: int | None = None
) -> BaseTaskConfig:
    # Replace the API key in the URL
    url = url.replace("{HALLUMINATE_API_KEY}", os.getenv("HALLUMINATE_API_KEY"))

    # Get the task start timestamp
    timestamp = int(datetime.now().timestamp()) if timestamp is None else timestamp

    # Create the user metadata
    user_metadata = UserMetadata(timestamp=timestamp, location=location, timezone=timezone)

    # Render the values (evaluate date formula)
    values = _render_values(values, timezone, timestamp)

    # Render the task (replace placeholders)
    task = _render_task(task, values, timestamp)

    # Create the eval config
    eval_target = get_import_path(HalluminateNoodleFlightsFlightSearchBudget)
    eval_config = {"_target_": eval_target, "task": task, "values": values, "user_metadata": user_metadata.model_dump()}

    # Create the task config
    return BaseTaskConfig(
        task=task,
        url=url,
        user_metadata=user_metadata,
        eval_config=eval_config,
    )


if __name__ == "__main__":
    from westworld.base import instantiate

    dataset_row = {
        "task_id": "halluminate_bench/noodle_flights/flight_search_budget/1",
        "task_generation_config_json": json.dumps(
            {
                "_target_": "halluminate_bench.noodle_flights.flight_search_budget.generate_task_config",
                "url": "https://flights.halluminate.ai/?apiKey={HALLUMINATE_API_KEY}",
                "task": (
                    "Are there any one-way flights from VCP to ORY on {date} that are less than $700? "
                    "If flight options are found, respond with: 'Yes, there is at least one option' followed by "
                    "the flight number and price of one option. If no options are found, respond with exactly: "
                    "'No options found'."
                ),
                "values": {
                    "origin": "VCP",
                    "destination": "ORY",
                    "date": "now() + timedelta(days=191)",
                    "budget": 700,
                },
                "location": "San Francisco, CA, United States",
                "timezone": "America/Los_Angeles",
                "timestamp": None,
            }
        ),
        "env": "sim",
        "domain": "noodle_flights",
        "l1_category": "travel",
        "l2_category": "flight_search_budget",
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
