import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import requests
from playwright.async_api import Page
from pydantic import BaseModel

from westworld.base import (
    BaseMetric,
    BaseTaskConfig,
    UserMetadata,
    async_retry_with_exponential_backoff,
    get_import_path,
)


# =========================
# Domain Models
# =========================


@dataclass
class ApplicationState:
    """Current state of the flight search application (from window.__testingAPI.getState())."""

    trip_type: str
    from_location: str
    to_location: str
    departure_date: str | None
    return_date: str | None
    travel_class: str
    is_search_ready: bool

    # Hints for city matching
    has_multiple_from_locations: bool
    has_multiple_to_locations: bool


@dataclass
class RoundtripFlightParams:
    """Expected flight search params from the task."""

    origin: str | None = None
    origin_airport: str | None = None
    origin_city: str | None = None
    origin_has_multiple_airports: bool | None = None

    destination: str | None = None
    destination_airport: str | None = None
    destination_city: str | None = None
    destination_has_multiple_airports: bool | None = None

    trip_type: str | None = None
    departure_date: str | None = None  # YYYY-MM-DD
    return_date: str | None = None  # YYYY-MM-DD
    travel_class: str | None = None


@dataclass
class VerificationResult:
    """Result of the verification process."""

    passed: bool
    details: dict[str, Any]
    message: str = ""


@dataclass(frozen=True)
class AirportRecord:
    """Airport record for task generation."""

    iata: str
    name: str
    city: str
    route_count: int
    country: str


# =========================
# Runtime Verification Functions
# =========================


def query_noodle_flights_simulator_roundtrip(
    origin: str, destination: str, departure_date: str, return_date: str, api_base_url: str | None = None
) -> dict[str, Any]:
    """
    Query the Noodle Flights simulator to get available roundtrip flights.

    Args:
        origin: Origin airport code (e.g., "MAN")
        destination: Destination airport code (e.g., "IAD")
        departure_date: Departure date in YYYY-MM-DD format
        return_date: Return date in YYYY-MM-DD format
        api_base_url: Optional base URL for the API

    Returns:
        Dictionary containing roundtrip flight data from the simulator
    """
    # Determine base URL for the Noodle Flights simulator
    provided_base = api_base_url or os.getenv("FLIGHTS_BASE_URL") or "https://flights.halluminate.ai"

    # Normalize to an API endpoint
    if provided_base.rstrip("/").endswith("/api/flights"):
        flights_endpoint = provided_base.rstrip("/")
    else:
        flights_endpoint = provided_base.rstrip("/") + "/api/flights"

    # Build query parameters for roundtrip
    params = {
        "origin": origin,
        "destination": destination,
        "departureDate": departure_date,
        "returnDate": return_date,
        "tripType": "round-trip",
        "travelClass": "economy",
        "adults": 1,
    }

    # Make the request
    api_url = f"{flights_endpoint}?{urlencode(params)}"
    try:
        response = requests.get(api_url, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to query flights API at {flights_endpoint}: {e}")


def find_cheapest_roundtrip_flights(flight_data: dict[str, Any]) -> dict[str, Any]:
    """
    Find the cheapest roundtrip flight combination from the simulator data.

    Args:
        flight_data: Flight data from the simulator API

    Returns:
        Dictionary containing cheapest roundtrip flight details
    """
    departing_flights = flight_data.get("departingFlights", [])
    returning_flights = flight_data.get("returningFlights", [])

    if not departing_flights or not returning_flights:
        return {"departureFlightNumbers": [], "returnFlightNumbers": [], "totalPrice": 0, "error": "No flights found"}

    # Find the cheapest combination
    cheapest_total = float("inf")
    cheapest_departure = None
    cheapest_return = None

    for dep_flight in departing_flights:
        for ret_flight in returning_flights:
            total_price = dep_flight.get("priceUsd", 0) + ret_flight.get("priceUsd", 0)
            if total_price < cheapest_total:
                cheapest_total = total_price
                cheapest_departure = dep_flight
                cheapest_return = ret_flight

    if cheapest_departure and cheapest_return:
        return {
            "departureFlightNumbers": [cheapest_departure.get("flightNumber", "UNKNOWN")],
            "returnFlightNumbers": [cheapest_return.get("flightNumber", "UNKNOWN")],
            "totalPrice": int(cheapest_total),
            "error": None,
        }
    else:
        return {
            "departureFlightNumbers": [],
            "returnFlightNumbers": [],
            "totalPrice": 0,
            "error": "Could not find cheapest combination",
        }


def get_runtime_ground_truth(task_data: dict[str, Any], api_base_url: str | None = None) -> dict[str, Any]:
    """
    Get ground truth by querying the simulator at runtime for roundtrip flights.

    Args:
        task_data: Task data containing origin, destination, departure_date, and return_date
        api_base_url: Optional base URL for the API

    Returns:
        Ground truth information from the simulator
    """
    values = task_data.get("values", {})
    origin = values.get("origin")
    destination = values.get("destination")
    departure_date = values.get("departureDate") or values.get("departure_date")
    return_date = values.get("returnDate") or values.get("return_date")

    if not all([origin, destination, departure_date, return_date]):
        raise ValueError("Task missing required fields: origin, destination, departure_date, or return_date")

    try:
        # Query the simulator for roundtrip flights
        flight_data = query_noodle_flights_simulator_roundtrip(
            origin, destination, departure_date, return_date, api_base_url
        )

        # Find the cheapest roundtrip combination
        cheapest_info = find_cheapest_roundtrip_flights(flight_data)

        return {
            "departureFlightNumbers": cheapest_info["departureFlightNumbers"],
            "returnFlightNumbers": cheapest_info["returnFlightNumbers"],
            "totalPrice": cheapest_info["totalPrice"],
            "totalDepartingFlights": len(flight_data.get("departingFlights", [])),
            "totalReturningFlights": len(flight_data.get("returningFlights", [])),
            "error": cheapest_info["error"],
        }
    except Exception as e:
        return {
            "departureFlightNumbers": [],
            "returnFlightNumbers": [],
            "totalPrice": 0,
            "totalDepartingFlights": 0,
            "totalReturningFlights": 0,
            "error": str(e),
        }


def normalize_price(price: str | int | float) -> str:
    """Convert to string and remove all non-numeric characters except decimal point."""
    return str(price).replace(",", "").replace("$", "").replace(" ", "")


def check_flight_numbers(response: str, flight_numbers: list[str]) -> bool:
    """Check if all flight numbers are found in the response."""
    for flight_num in flight_numbers:
        if "/" in flight_num:
            # Split combined flight numbers and check each one
            individual_flights = flight_num.split("/")
            found_all = True
            for individual_flight in individual_flights:
                individual_flight = individual_flight.strip()
                if not (
                    individual_flight in response or f"{individual_flight[:2]} {individual_flight[2:]}" in response
                ):
                    found_all = False
                    break
            if not found_all:
                return False
        else:
            # Handle single flight numbers
            if not (flight_num in response or f"{flight_num[:2]} {flight_num[2:]}" in response):
                return False
    return True


def verify_agent_response(
    agent_response: str, task_id: str, task_data: dict[str, Any], *, base_url: str | None = None
) -> VerificationResult:
    """
    Verify an agent's response to a roundtrip flight search task using runtime API data.

    Args:
        agent_response: The agent's response (should contain flight numbers and total price)
        task_id: The task ID
        task_data: Task data containing origin, destination, departure_date, and return_date
        base_url: Optional base URL for the simulator

    Returns:
        Verification result
    """
    # Get ground truth from simulator
    ground_truth = get_runtime_ground_truth(task_data, base_url)

    # Check for all flight numbers in both directions
    has_departure_flights = check_flight_numbers(agent_response, ground_truth["departureFlightNumbers"])
    has_return_flights = check_flight_numbers(agent_response, ground_truth["returnFlightNumbers"])

    # Extract all numbers from response and normalize them
    numbers = re.findall(r"[\d,]+(?:\.\d+)?", agent_response) or []
    normalized_numbers = [normalize_price(num) for num in numbers]
    normalized_truth_price = normalize_price(ground_truth["totalPrice"])

    # Check if any of the normalized numbers match our total price
    has_price = normalized_truth_price in normalized_numbers

    return VerificationResult(
        passed=has_departure_flights and has_return_flights and has_price,
        details={
            "departureFlightsFound": has_departure_flights,
            "returnFlightsFound": has_return_flights,
            "priceFound": has_price,
            "expectedDepartureFlights": ground_truth["departureFlightNumbers"],
            "expectedReturnFlights": ground_truth["returnFlightNumbers"],
            "expectedTotalPrice": ground_truth["totalPrice"],
            "numbersFound": numbers,
            "totalDepartingFlights": ground_truth.get("totalDepartingFlights", 0),
            "totalReturningFlights": ground_truth.get("totalReturningFlights", 0),
            "error": ground_truth.get("error"),
        },
        message=(
            "Correct! Found all flight numbers and total price."
            if has_departure_flights and has_return_flights and has_price
            else f"Missing {'departure flight number(s)' if not has_departure_flights else ''}"
            f"{' and ' if not has_departure_flights and not has_return_flights else ''}"
            f"{'return flight number(s)' if not has_return_flights else ''}"
            f"{' and ' if (not has_departure_flights or not has_return_flights) and not has_price else ''}"
            f"{'total price' if not has_price else ''} in response."
        ),
    )


# =========================
# UI State Verification Functions
# =========================


def is_specified(value: Any) -> bool:
    """Check if a value was specified (not None or empty string)."""
    if value is None:
        return False
    if isinstance(value, str) and value.strip() == "":
        return False
    return True


def normalize_string_for_comparison(text: str | None) -> str:
    """Normalize string for case-insensitive comparison."""
    return (text or "").strip().lower()


def dates_match(date1: str | None, date2: str | None) -> bool:
    """Compare two dates in ISO format (YYYY-MM-DD)."""
    try:
        # Parse dates and compare only the date part
        d1 = datetime.fromisoformat(str(date1).split("T")[0])
        d2 = datetime.fromisoformat(str(date2).split("T")[0])
        return d1.date() == d2.date()
    except (ValueError, AttributeError):
        return False


def bidirectional_substring_match(expected: str | None, actual: str | None) -> bool:
    e = normalize_string_for_comparison(expected)
    a = normalize_string_for_comparison(actual)
    return (e in a) or (a in e)


def location_matches_with_sophisticated_fallback(
    expected_code: str | None,
    expected_airport: str | None,
    expected_city: str | None,
    has_multiple_airports: bool | None,
    actual: str | None,
) -> bool:
    # 1) airport code
    if expected_code and bidirectional_substring_match(expected_code, actual):
        return True
    # 2) airport name
    if expected_airport and bidirectional_substring_match(expected_airport, actual):
        return True
    # 3) multiple-airport cities require code/airport (reject plain city)
    if has_multiple_airports:
        return False
    # 4) otherwise allow city name match
    if expected_city and bidirectional_substring_match(expected_city, actual):
        return True
    return False


def verify_flight_search_roundtrip_complete(
    params: RoundtripFlightParams,
    state: ApplicationState,
    agent_response: str | None = None,
    *,
    base_url: str | None = None,
) -> VerificationResult:
    """
    Verify that the application state matches the expected flight search parameters,
    and validate the agent's flight numbers and total price response.

    Args:
        params: Expected parameters from the task
        state: Current application state
        agent_response: agent response (flight numbers and total price) to validate
        base_url: Optional base URL for the flights API passed to simulator queries.

    Returns:
        VerificationResult with combined UI state and agent response validation results
    """
    errors: list[str] = []

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
    if is_specified(params.departure_date):
        if not state.departure_date:
            errors.append("Departure date not set")
        elif not dates_match(params.departure_date, state.departure_date):
            errors.append(f"Departure date mismatch. Expected {params.departure_date}, found {state.departure_date}")

    # Check return date if specified
    if is_specified(params.return_date):
        if not state.return_date:
            errors.append("Return date not set")
        elif not dates_match(params.return_date, state.return_date):
            errors.append(f"Return date mismatch. Expected {params.return_date}, found {state.return_date}")

    # Check travel class if specified
    if is_specified(params.travel_class):
        if normalize_string_for_comparison(state.travel_class) != normalize_string_for_comparison(params.travel_class):
            errors.append(f'Travel class mismatch. Expected "{params.travel_class}", found "{state.travel_class}"')

    # Check if the form is ready
    if not state.is_search_ready:
        errors.append("Search form is not complete - missing required fields")

    # RUNTIME VERIFICATION: Query the simulator to get real roundtrip flight data
    flight_info = {}
    if params.origin and params.destination and params.departure_date and params.return_date:
        try:
            # Create task_data structure from params for simulator query
            task_data = {
                "values": {
                    "origin": params.origin,
                    "destination": params.destination,
                    "departureDate": params.departure_date,
                    "returnDate": params.return_date,
                }
            }

            # Query the simulator
            ground_truth = get_runtime_ground_truth(task_data)

            flight_info = {
                "runtime_verification": True,
                "simulator_total_departing_flights": ground_truth.get("totalDepartingFlights", 0),
                "simulator_total_returning_flights": ground_truth.get("totalReturningFlights", 0),
                "simulator_cheapest_departure_flights": ground_truth.get("departureFlightNumbers"),
                "simulator_cheapest_return_flights": ground_truth.get("returnFlightNumbers"),
                "simulator_cheapest_total_price": ground_truth.get("totalPrice"),
                "simulator_error": ground_truth.get("error") if ground_truth.get("error") else None,
            }

        except Exception as e:
            flight_info = {"runtime_verification": False, "simulator_error": str(e)}
    else:
        flight_info = {
            "runtime_verification": False,
            "simulator_error": "Missing required fields for flight validation",
        }

    # PROGRAMMATIC AGENT RESPONSE VALIDATION
    agent_validation = {}
    agent_passed = True

    if len(errors) == 0:  # Only validate if UI is correct
        if agent_response is not None:
            try:
                # Create task_data structure from params for simulator query
                task_data = {
                    "values": {
                        "origin": params.origin,
                        "destination": params.destination,
                        "departureDate": params.departure_date,
                        "returnDate": params.return_date,
                    }
                }

                # Validate the agent response using runtime API
                if flight_info.get("runtime_verification"):
                    agent_result = verify_agent_response(agent_response, "task", task_data, base_url=base_url)

                    agent_validation = {
                        "agent_response": agent_response,
                        "departure_flights_found": agent_result.details.get("departureFlightsFound"),
                        "return_flights_found": agent_result.details.get("returnFlightsFound"),
                        "price_found": agent_result.details.get("priceFound"),
                        "is_correct": agent_result.passed,
                    }
                    agent_passed = agent_result.passed
                else:
                    agent_validation = {
                        "agent_response": agent_response,
                        "error": "Flight validation failed, cannot verify agent response",
                        "is_correct": False,
                    }
                    agent_passed = False
            except Exception as e:
                agent_validation = {"error": f"Agent validation failed: {str(e)}", "is_correct": False}
                agent_passed = False
        else:
            # No agent response provided; fail agent validation
            agent_validation = {"error": "No agent response provided", "is_correct": False}
            agent_passed = False

    # CRITICAL: Determine overall result based on BOTH UI state AND agent response
    ui_passed = len(errors) == 0
    overall_passed = ui_passed and agent_passed

    # Build actual state summary for debugging
    actual_summary: dict[str, Any] = {
        "trip_type": state.trip_type,
        "from_location": state.from_location,
        "to_location": state.to_location,
        "departure_date": state.departure_date,
        "return_date": state.return_date,
        "travel_class": state.travel_class,
        "is_search_ready": state.is_search_ready,
        "agent_validation": agent_validation,
    }

    # Create a cleaner expected summary that only shows specified fields
    expected_summary = {
        "origin": params.origin,
        "destination": params.destination,
        "departure_date": params.departure_date,
        "return_date": params.return_date,
        "trip_type": "round-trip",
        "travel_class": "economy",
        "expected_departure_flights": flight_info.get("simulator_cheapest_departure_flights"),
        "expected_return_flights": flight_info.get("simulator_cheapest_return_flights"),
        "expected_total_price": flight_info.get("simulator_cheapest_total_price"),
    }

    # Create comprehensive message that reflects the combined result
    if overall_passed:
        if flight_info.get("runtime_verification"):
            message = (
                f"✅ OVERALL: PASS | UI State: PASS | Flight Validation: "
                f"{flight_info.get('simulator_total_departing_flights')} departing "
                f"+ {flight_info.get('simulator_total_returning_flights')} returning flights found, "
                f"cheapest {flight_info.get('simulator_cheapest_departure_flights')} + "
                f"{flight_info.get('simulator_cheapest_return_flights')} at "
                f"${flight_info.get('simulator_cheapest_total_price')} | Agent Response: ✅ PASS"
            )
        else:
            message = (
                f"✅ OVERALL: PASS | UI State: PASS | Flight Validation: ⚠️ "
                f"{flight_info.get('simulator_error')} | Agent Response: ✅ PASS"
            )
    else:
        # Determine what failed
        if not ui_passed:
            message = (
                f"❌ OVERALL: FAIL | UI State: FAIL ({len(errors)} errors) | Agent Response: "
                f"{'PASS' if agent_passed else 'FAIL'}"
            )
        elif not agent_passed:
            if agent_response is None:
                message = "❌ OVERALL: FAIL | UI State: PASS | Agent Response: FAIL - No agent response provided"
            else:
                message = (
                    f"❌ OVERALL: FAIL | UI State: PASS | Agent Response: FAIL - "
                    f"{agent_validation.get('message', 'Incorrect answer')}"
                )
        else:
            message = "❌ OVERALL: FAIL | UI State: PASS | Agent Response: PASS (unexpected failure)"

    return VerificationResult(
        passed=overall_passed,
        details={"errors": errors, "expected": expected_summary, "actual": actual_summary},
        message=message,
    )


# =========================
# Factory Functions
# =========================


def create_application_state_from_dict(data: dict[str, Any]) -> ApplicationState:
    """
    Map window.__testingAPI.getState() to ApplicationState.

    Note: The testing API only returns form state, not search results.
    Flight results are validated separately via runtime verification against the simulator API.
    """
    trip_type = data.get("tripType", "") or data.get("trip_type", "")
    from_loc = data.get("fromLocation", "") or data.get("from_location", "")
    to_loc = data.get("toLocation", "") or data.get("to_location", "")
    dep_date = data.get("departureDate") or data.get("departure_date")
    ret_date = data.get("returnDate") or data.get("return_date")
    tclass = data.get("travelClass", "") or data.get("travel_class", "")
    ready = bool(data.get("isSearchReady", False))

    return ApplicationState(
        trip_type=trip_type,
        from_location=from_loc,
        to_location=to_loc,
        departure_date=dep_date,
        return_date=ret_date,
        travel_class=tclass,
        is_search_ready=ready,
        has_multiple_from_locations=bool(data.get("hasMultipleFromLocations", False)),
        has_multiple_to_locations=bool(data.get("hasMultipleToLocations", False)),
    )


def create_flight_search_roundtrip_params_from_dict(data: dict[str, Any]) -> RoundtripFlightParams:
    """Create a RoundtripFlightParams object from a dictionary."""
    return RoundtripFlightParams(
        origin=data.get("origin"),
        origin_airport=data.get("originAirport"),
        origin_city=data.get("originCity"),
        origin_has_multiple_airports=data.get("originHasMultipleAirports"),
        destination=data.get("destination"),
        destination_airport=data.get("destinationAirport"),
        destination_city=data.get("destinationCity"),
        destination_has_multiple_airports=data.get("destinationHasMultipleAirports"),
        trip_type=data.get("tripType"),
        departure_date=data.get("departureDate") or data.get("departure_date"),
        return_date=data.get("returnDate") or data.get("return_date"),
        travel_class=data.get("class") or data.get("travelClass"),
    )


# =========================
# Evaluator (BaseMetric Implementation)
# =========================


class FinalResult(BaseModel):
    """Result model for Noodle Flights flight search roundtrip task."""

    score: float
    reasoning: str
    current_state: dict[str, Any]
    ground_truth: RoundtripFlightParams


class HalluminateNoodleFlightsFlightSearchRoundtrip(BaseMetric):
    """Evaluator that runs verifier for Noodle Flights flight search roundtrip task."""

    def __init__(self, task: str, values: dict[str, Any], user_metadata: UserMetadata) -> None:
        super().__init__()
        self.task = task
        self.values = values
        self.user_metadata = user_metadata

        self._ground_truth = create_flight_search_roundtrip_params_from_dict(values)
        self._page: Page | None = None
        self._answer_message: str | None = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(task={self.task}, values={self.values}, user_metadata={self.user_metadata})"

    async def reset(self) -> None:
        """Reset the evaluator state."""
        self._page = None
        self._answer_message = None

    async def update(self, /, **kwargs):
        self._page = kwargs.get("page")
        self._answer_message = kwargs.get("answer_message")

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
        result = verify_flight_search_roundtrip_complete(
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
# Task Generation Functions
# =========================


def _evaluate_date_formula(formula_str: str, timezone: str, timestamp: int) -> str:
    """Safely evaluate a date formula string and return YYYY-MM-DD string.

    Example:
        Input:  formula_str="now() + timedelta(days=1)", timezone="America/Los_Angeles", timestamp=1715702400
        Output: "2024-05-15"

        Input:  formula_str="now() + timedelta(days=1)", timezone="Asia/Tokyo", timestamp=1715702400
        Output: "2024-05-16"
    """

    def now():
        return datetime.fromtimestamp(timestamp, tz=ZoneInfo(timezone))

    safe_globals = {"timedelta": timedelta, "now": now}
    result = eval(formula_str, safe_globals)
    return result.strftime("%Y-%m-%d")


def _format_date_for_display(date_str: str) -> str:
    """Format YYYY-MM-DD date string to human-readable format for task statement.

    Example:
        Input:  "2026-01-15"
        Output: "January 15, 2026"
    """
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%B %d, %Y")
    except (ValueError, AttributeError):
        # If parsing fails, return original string
        return date_str


def _render_values(values: dict[str, Any], timezone: str, timestamp: int) -> dict[str, Any]:
    """Evaluate date formulas in the departureDate and returnDate fields for flight-search-roundtrip tasks.

    Converts formula strings to YYYY-MM-DD format strings that can be used by the verifier.
    """
    evaluated = dict(values)

    # Evaluate departureDate if it's a formula
    departure_date_value = evaluated.get("departureDate")
    if isinstance(departure_date_value, str) and (
        "now()" in departure_date_value or "timedelta" in departure_date_value
    ):
        evaluated["departureDate"] = _evaluate_date_formula(departure_date_value, timezone, timestamp)

    # Evaluate returnDate if it's a formula
    return_date_value = evaluated.get("returnDate")
    if isinstance(return_date_value, str) and ("now()" in return_date_value or "timedelta" in return_date_value):
        evaluated["returnDate"] = _evaluate_date_formula(return_date_value, timezone, timestamp)

    return evaluated


def _render_task(statement: str, values: dict[str, Any]) -> str:
    """Replace {departureDate} and {returnDate} placeholders with formatted dates."""
    result = statement

    # Replace {departureDate} placeholder
    if "{departureDate}" in result:
        departure_date = values.get("departureDate")
        if isinstance(departure_date, str):
            # Format from YYYY-MM-DD to "January 01, 2025" format
            formatted_date = _format_date_for_display(departure_date)
            result = result.replace("{departureDate}", formatted_date)

    # Replace {returnDate} placeholder
    if "{returnDate}" in result:
        return_date = values.get("returnDate")
        if isinstance(return_date, str):
            # Format from YYYY-MM-DD to "January 01, 2025" format
            formatted_date = _format_date_for_display(return_date)
            result = result.replace("{returnDate}", formatted_date)

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

    # Render the values (evaluate date formulas)
    values = _render_values(values, timezone, timestamp)

    # Render the task (replace placeholders)
    task = _render_task(task, values)

    # Create the eval config
    eval_target = get_import_path(HalluminateNoodleFlightsFlightSearchRoundtrip)
    eval_config = {"_target_": eval_target, "task": task, "values": values, "user_metadata": user_metadata.model_dump()}

    # Create the task config
    return BaseTaskConfig(
        task=task,
        url=url,
        user_metadata=user_metadata,
        eval_config=eval_config,
    )


if __name__ == "__main__":
    import json

    from westworld.base import DatasetItem, instantiate

    dataset_row = {
        "task_id": "halluminate_bench/noodle_flights/flight_search_roundtrip/1",
        "task_generation_config_json": json.dumps(
            {
                "_target_": "halluminate_bench.noodle_flights.flight_search_roundtrip.generate_task_config",
                "url": "https://flights.halluminate.ai/?apiKey={HALLUMINATE_API_KEY}",
                "task": (
                    "Find the cheapest round-trip flight option from Accra (ACC) to Philadelphia (PHL) if I leave on "
                    "{departureDate} and come back on {returnDate}. Return the total price and the flight numbers for "
                    "all flights."
                ),
                "values": {
                    "origin": "ACC",
                    "originAirport": "Kotoka International Airport",
                    "originCity": "Accra",
                    "originCountry": "Ghana",
                    "destination": "PHL",
                    "destinationAirport": "Philadelphia International Airport",
                    "destinationCity": "Philadelphia",
                    "destinationCountry": "United States",
                    "departureDate": "now() + timedelta(days=158)",
                    "returnDate": "now() + timedelta(days=158) + timedelta(days=5)",
                },
                "location": "San Francisco, CA, United States",
                "timezone": "America/Los_Angeles",
                "timestamp": None,
            }
        ),
        "env": "sim",
        "domain": "noodle_flights",
        "l1_category": "travel",
        "l2_category": "flight_search_roundtrip",
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
