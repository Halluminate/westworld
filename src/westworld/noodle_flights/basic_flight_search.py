import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Dict, List, Optional, TypedDict
from zoneinfo import ZoneInfo

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
# Core domain models
# =========================


@dataclass
class PassengerCounts:
    """Passenger counts for the flight search."""

    adults: int
    children: int
    infants: int
    lap_infants: int


@dataclass
class FlightSegment:
    """Flight segment for multi-city trips."""

    id: str
    from_location: str
    to_location: str
    date: Optional[str] = None


@dataclass
class ApplicationState:
    """Current state of the flight search application."""

    trip_type: str
    from_location: str
    to_location: str
    passengers: PassengerCounts
    total_passengers: int
    travel_class: str
    flight_segments: List[FlightSegment]
    is_search_ready: bool
    has_multiple_from_locations: bool
    has_multiple_to_locations: bool
    departure_date: Optional[str] = None
    return_date: Optional[str] = None


@dataclass
class FlightSearchParams:
    """Expected flight search parameters from the task."""

    origin: Optional[str] = None
    origin_airport: Optional[str] = None
    origin_city: Optional[str] = None
    origin_has_multiple_airports: Optional[bool] = None
    destination: Optional[str] = None
    destination_airport: Optional[str] = None
    destination_city: Optional[str] = None
    destination_has_multiple_airports: Optional[bool] = None
    trip_type: Optional[str] = None
    departure_date: Optional[str] = None
    return_date: Optional[str] = None
    adults: Optional[int] = None
    children: Optional[int] = None
    infants: Optional[int] = None
    lap_infants: Optional[int] = None
    travel_class: Optional[str] = None
    flight_segments: Optional[List[Dict[str, Any]]] = None


@dataclass
class VerificationResult:
    """Result of the verification process."""

    passed: bool
    details: Dict[str, Any]


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


def normalize_string_for_comparison(text: str) -> str:
    """Normalize string for case-insensitive comparison."""
    return text.lower().strip()


def dates_match(date1: str, date2: str) -> bool:
    """Compare two dates in ISO format (YYYY-MM-DD)."""
    try:
        # Parse dates and compare only the date part
        d1 = datetime.fromisoformat(date1.split("T")[0])
        d2 = datetime.fromisoformat(date2.split("T")[0])
        return d1.date() == d2.date()
    except (ValueError, AttributeError):
        return False


def bidirectional_substring_match(expected: str, actual: str) -> bool:
    """Check if two strings match using bidirectional substring matching."""
    expected_norm = normalize_string_for_comparison(expected)
    actual_norm = normalize_string_for_comparison(actual)

    # Check if either string contains the other
    return expected_norm in actual_norm or actual_norm in expected_norm


def location_matches_with_sophisticated_fallback(
    expected_code: str, expected_airport: str, expected_city: str, has_multiple_airports: bool, actual: str
) -> bool:
    """
    Check if location matches with sophisticated fallback logic.

    Logic:
    1. Always try to match airport code first
    2. Always try to match airport name
    3. For cities with multiple airports: only accept code/airport name matches
    4. For cities with single airport: also accept city name matches

    Args:
        expected_code: Expected airport code (e.g., "LAX")
        expected_airport: Expected airport name (e.g., "Los Angeles International Airport")
        expected_city: Expected city name (e.g., "Los Angeles")
        has_multiple_airports: Whether the city has multiple airports
        actual: Actual location string from the application

    Returns:
        True if there's a match, False otherwise
    """

    # 1. Always try to match airport code first
    if bidirectional_substring_match(expected_code, actual):
        return True

    # 2. Always try to match airport name
    if bidirectional_substring_match(expected_airport, actual):
        return True

    # 3. For cities with multiple airports: only accept code/airport name matches (already checked above)
    if has_multiple_airports:
        return False

    # 4. For cities with single airport: also accept city name matches
    if bidirectional_substring_match(expected_city, actual):
        return True

    return False


def verify_flight_search_complete(state: ApplicationState, params: FlightSearchParams) -> VerificationResult:
    """
    Verify that the application state matches the expected flight search parameters.

    Args:
        state: Current application state
        params: Expected parameters from the task

    Returns:
        VerificationResult with pass/fail status and details
    """
    errors: List[str] = []

    # Check trip type if specified
    if is_specified(params.trip_type):
        if state.trip_type != params.trip_type:
            errors.append(f'Trip type mismatch. Expected "{params.trip_type}", found "{state.trip_type}"')

    # For multi-city trips, validate segments
    if state.trip_type == "multi-city" and is_specified(params.flight_segments):
        # Check each specified segment
        for index, expected_segment in enumerate(params.flight_segments):
            if index >= len(state.flight_segments):
                errors.append(f"Missing flight segment {index + 1}")
                continue

            actual_segment = state.flight_segments[index]

            if expected_segment.get("date"):
                expected_date = expected_segment["date"]
                if not dates_match(expected_date, actual_segment.date or ""):
                    errors.append(
                        f"Segment {index + 1} date mismatch. Expected {expected_date}, found {actual_segment.date}"
                    )

    else:
        # For non-multi-city trips, check origin/destination/dates

        # Check origin if specified
        if is_specified(params.origin):
            # Skip validation if origin is empty (allows for tasks where origin is not specified)
            origin_matches = location_matches_with_sophisticated_fallback(
                params.origin,
                params.origin_airport,
                params.origin_city,
                params.origin_has_multiple_airports,
                state.from_location,
            )
            if not origin_matches:
                expected_display = f'"{params.origin}"'
                if params.origin_airport:
                    expected_display += f' or "{params.origin_airport}"'
                if params.origin_city and not params.origin_has_multiple_airports:
                    expected_display += f' or "{params.origin_city}"'
                errors.append(f'Origin location mismatch. Expected {expected_display}, found "{state.from_location}"')

        # Check destination if specified
        if is_specified(params.destination):
            # Skip validation if destination is empty (allows for tasks where destination is not specified)
            destination_matches = location_matches_with_sophisticated_fallback(
                params.destination,
                params.destination_airport,
                params.destination_city,
                params.destination_has_multiple_airports,
                state.to_location,
            )
            if not destination_matches:
                if not destination_matches:
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
            else:
                if not dates_match(params.departure_date, state.departure_date):
                    errors.append(
                        f"Departure date mismatch. Expected {params.departure_date}, found {state.departure_date}"
                    )

        # Check return date if specified (and trip type is round-trip)
        if is_specified(params.return_date):
            if state.trip_type == "round-trip":
                if not state.return_date:
                    errors.append("Return date not set for round-trip")
                else:
                    if not dates_match(params.return_date, state.return_date):
                        errors.append(f"Return date mismatch. Expected {params.return_date}, found {state.return_date}")

    # Check passengers if specified
    if is_specified(params.adults):
        if state.passengers.adults != params.adults:
            errors.append(f"Adults count mismatch. Expected {params.adults}, found {state.passengers.adults}")

    if is_specified(params.children):
        if state.passengers.children != params.children:
            errors.append(f"Children count mismatch. Expected {params.children}, found {state.passengers.children}")

    if is_specified(params.infants):
        if state.passengers.infants != params.infants:
            errors.append(
                f"Infants (in seat) count mismatch. Expected {params.infants}, found {state.passengers.infants}"
            )

    if is_specified(params.lap_infants):
        if state.passengers.lap_infants != params.lap_infants:
            errors.append(
                f"Infants (on lap) count mismatch. Expected {params.lap_infants}, found {state.passengers.lap_infants}"
            )

    # Check travel class if specified
    if is_specified(params.travel_class):
        if state.travel_class != params.travel_class:
            errors.append(f'Travel class mismatch. Expected "{params.travel_class}", found "{state.travel_class}"')

    # Always check if form is ready to search
    if not state.is_search_ready:
        errors.append("Search form is not complete - missing required fields")

    # Build actual state summary for debugging
    actual_summary = {
        "trip_type": state.trip_type,
        "from_location": state.from_location,
        "to_location": state.to_location,
        "departure_date": state.departure_date,
        "return_date": state.return_date,
        "passengers": {
            "adults": state.passengers.adults,
            "children": state.passengers.children,
            "infants": state.passengers.infants,
            "lap_infants": state.passengers.lap_infants,
        },
        "travel_class": state.travel_class,
        "is_search_ready": state.is_search_ready,
    }

    if state.trip_type == "multi-city":
        actual_summary["flight_segments"] = [
            {"id": seg.id, "from": seg.from_location, "to": seg.to_location, "date": seg.date}
            for seg in state.flight_segments
        ]

    return VerificationResult(
        passed=len(errors) == 0, details={"errors": errors, "expected": params.__dict__, "actual": actual_summary}
    )


def create_application_state_from_dict(data: Dict[str, Any]) -> ApplicationState:
    """Create an ApplicationState object from a dictionary."""
    passengers_data = data.get("passengers", {})
    passengers = PassengerCounts(
        adults=passengers_data.get("adults", 0),
        children=passengers_data.get("children", 0),
        infants=passengers_data.get("infants", 0),
        lap_infants=passengers_data.get("lapInfants", 0),
    )

    flight_segments = []
    for seg_data in data.get("flightSegments", []):
        segment = FlightSegment(
            id=seg_data.get("id", ""),
            from_location=seg_data.get("from", ""),
            to_location=seg_data.get("to", ""),
            date=seg_data.get("date"),
        )
        flight_segments.append(segment)

    return ApplicationState(
        trip_type=data.get("tripType", ""),
        from_location=data.get("fromLocation", ""),
        to_location=data.get("toLocation", ""),
        departure_date=data.get("departureDate"),
        return_date=data.get("returnDate"),
        passengers=passengers,
        total_passengers=data.get("totalPassengers", 0),
        travel_class=data.get("travelClass", ""),
        flight_segments=flight_segments,
        is_search_ready=data.get("isSearchReady", False),
        has_multiple_from_locations=data.get("hasMultipleFromLocations", False),
        has_multiple_to_locations=data.get("hasMultipleToLocations", False),
    )


def create_flight_search_params_from_dict(data: Dict[str, Any]) -> FlightSearchParams:
    """Create a FlightSearchParams object from a dictionary."""
    return FlightSearchParams(
        origin=data.get("origin"),
        origin_airport=data.get("originAirport"),
        origin_city=data.get("originCity"),
        origin_has_multiple_airports=data.get("originHasMultipleAirports"),
        destination=data.get("destination"),
        destination_airport=data.get("destinationAirport"),
        destination_city=data.get("destinationCity"),
        destination_has_multiple_airports=data.get("destinationHasMultipleAirports"),
        trip_type=data.get("tripType"),
        departure_date=data.get("departureDate"),
        return_date=data.get("returnDate"),
        adults=data.get("adults"),
        children=data.get("children"),
        infants=data.get("infants"),
        lap_infants=data.get("lapInfants"),
        travel_class=data.get("class"),
        flight_segments=data.get("flightSegments"),
    )


# =========================
# Evaluator
# =========================


class InputDict(TypedDict, total=False):
    page: "Page"


class FinalResult(BaseModel):
    """Result model for Noodle Flights basic flight search task."""

    score: float
    reasoning: str
    current_state: dict[str, Any]
    ground_truth: FlightSearchParams


class HalluminateNoodleFlightsBasicFlightSearch(BaseMetric):
    """Evaluator that runs verifier for Noodle Flights basic flight search task."""

    def __init__(self, task: str, values: dict[str, Any], user_metadata: UserMetadata) -> None:
        super().__init__()
        self.task = task
        self.values = values
        self.user_metadata = user_metadata

        self._ground_truth = create_flight_search_params_from_dict(values)
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
        result = verify_flight_search_complete(state_obj, self._ground_truth)

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
        Input:  formula_str="now() + timedelta(days=120)", timezone="America/Los_Angeles", timestamp=1715702400
        Output: datetime object

        Input:  formula_str="now() + timedelta(days=120)", timezone="Asia/Tokyo", timestamp=1715702400
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
    """Render departureDate/returnDate formulas for basic-flight-search tasks.

    Converts formula strings to YYYY-MM-DD format strings (what verifier expects).
    All other fields are left unchanged.
    """
    evaluated = dict(values)

    # Evaluate departureDate if it's a formula
    dep_date = evaluated.get("departureDate")
    if isinstance(dep_date, str) and ("now()" in dep_date or "timedelta" in dep_date):
        dep_dt = _evaluate_date_formula(dep_date, timezone, timestamp)
        evaluated["departureDate"] = dep_dt.strftime("%Y-%m-%d")

    # Evaluate returnDate if it's a formula (only for round-trip)
    ret_date = evaluated.get("returnDate")
    if isinstance(ret_date, str) and ("now()" in ret_date or "timedelta" in ret_date):
        ret_dt = _evaluate_date_formula(ret_date, timezone, timestamp)
        evaluated["returnDate"] = ret_dt.strftime("%Y-%m-%d")

    return evaluated


def _render_task(task: str, values: dict[str, Any], timestamp: int) -> str:
    """Replace {departureDate}/{returnDate} placeholders with formatted dates.

    Example:
        Input task: "Book a flight departing on {departureDate}..."
        Input values: {"departureDate": "2025-12-04"}
        Output: "Book a flight departing on December 4, 2025..."
    """
    result = task

    # Replace {departureDate} placeholder
    if "{departureDate}" in result:
        dep_date = values.get("departureDate")
        if isinstance(dep_date, str) and len(dep_date) == 10:  # YYYY-MM-DD format
            try:
                dep_dt = datetime.strptime(dep_date, "%Y-%m-%d")
                result = result.replace("{departureDate}", _format_date_for_task(dep_dt))
            except ValueError:
                # If parsing fails, leave placeholder as-is
                pass

    # Replace {returnDate} placeholder
    if "{returnDate}" in result:
        ret_date = values.get("returnDate")
        if isinstance(ret_date, str) and len(ret_date) == 10:  # YYYY-MM-DD format
            try:
                ret_dt = datetime.strptime(ret_date, "%Y-%m-%d")
                result = result.replace("{returnDate}", _format_date_for_task(ret_dt))
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

    # Render the values (evaluate date formulas)
    values = _render_values(values, timezone, timestamp)

    # Render the task (replace placeholders)
    task = _render_task(task, values, timestamp)

    # Create the eval config
    eval_target = get_import_path(HalluminateNoodleFlightsBasicFlightSearch)
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
        "task_id": "halluminate_bench/noodle_flights/basic_flight_search/1",
        "task_generation_config_json": json.dumps(
            {
                "_target_": "halluminate_bench.noodle_flights.basic_flight_search.generate_task_config",
                "url": "https://flights.halluminate.ai/?apiKey={HALLUMINATE_API_KEY}",
                "task": (
                    "Book a one-way economy class flight for 1 adult passenger from RIC to JFK "
                    "departing on {departureDate}. Submit the search to complete the task."
                ),
                "values": {
                    "origin": "RIC",
                    "originAirport": "Richmond International Airport",
                    "originCity": "Richmond",
                    "originCountry": "United States",
                    "originHasMultipleAirports": True,
                    "destination": "JFK",
                    "destinationAirport": "John F Kennedy International Airport",
                    "destinationCity": "New York",
                    "destinationCountry": "United States",
                    "destinationHasMultipleAirports": True,
                    "tripType": "one-way",
                    "departureDate": "now() + timedelta(days=309)",
                    "adults": 1,
                    "class": "economy",
                },
                "location": "San Francisco, CA, United States",
                "timezone": "America/Los_Angeles",
                "timestamp": None,
            }
        ),
        "env": "sim",
        "domain": "noodle_flights",
        "l1_category": "travel",
        "l2_category": "basic_flight_search",
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
