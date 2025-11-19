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


@dataclass
class RoomDetails:
    """Room details for hotel search."""

    adults: int
    children: int
    child_ages: List[int]


@dataclass
class ApplicationState:
    """Current state of the hotel search application."""

    destination: str
    checkin: Optional[str] = None
    checkout: Optional[str] = None
    # Legacy aggregated fields (for backwards compatibility)
    adults: int = 0
    children: int = 0
    child_ages: List[int] = None
    rooms: int = 0
    # New detailed room information
    roomDetails: List[RoomDetails] = None
    addFlight: bool = False
    addCar: bool = False
    isSubmitted: bool = False

    def __post_init__(self):
        if self.child_ages is None:
            self.child_ages = []
        if self.roomDetails is None:
            self.roomDetails = []


@dataclass
class HotelSearchParams:
    """Expected hotel search parameters from the task."""

    city: Optional[str] = None
    checkin: Optional[str] = None
    checkout: Optional[str] = None
    # Legacy single-room fields (for backwards compatibility)
    adults: Optional[int] = None
    children: Optional[int] = None
    child_ages: Optional[List[int]] = None
    # New multi-room and additional fields
    rooms: Optional[List[Dict[str, Any]]] = None
    addFlight: Optional[bool] = None
    addCar: Optional[bool] = None


@dataclass
class VerificationResult:
    """Result of the verification process."""

    passed: bool
    details: Dict[str, Any]


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


def verify_hotel_search_complete(state: ApplicationState, params: HotelSearchParams) -> VerificationResult:
    """
    Verify that the application state matches the expected hotel search parameters.

    Args:
        state: Current application state
        params: Expected parameters from the task

    Returns:
        VerificationResult with pass/fail status and details
    """
    errors: List[str] = []

    # If no parameters specified, everything passes
    params_dict = {k: v for k, v in params.__dict__.items() if is_specified(v)}
    if not params_dict:
        return VerificationResult(
            passed=True, details={"expected": params.__dict__, "actual": state.__dict__, "errors": errors}
        )

    # Only check fields that were specified in params
    if is_specified(params.city):
        if not state.destination or normalize_string_for_comparison(params.city) not in normalize_string_for_comparison(
            state.destination
        ):
            errors.append(f'Destination mismatch: expected "{params.city}" to be in "{state.destination}"')

    if is_specified(params.checkin):
        if not dates_match(params.checkin, state.checkin or ""):
            errors.append(f'Check-in date mismatch: expected "{params.checkin}", got "{state.checkin}"')

    if is_specified(params.checkout):
        if not dates_match(params.checkout, state.checkout or ""):
            errors.append(f'Check-out date mismatch: expected "{params.checkout}", got "{state.checkout}"')

    # Handle room verification - support both legacy and new formats
    if is_specified(params.rooms):
        # New multi-room format specified
        if len(state.roomDetails) != len(params.rooms):
            errors.append(f"Room count mismatch: expected {len(params.rooms)} rooms, got {len(state.roomDetails)}")
        else:
            # Verify room configurations as a set (order doesn't matter)
            expected_rooms = list(params.rooms)
            actual_rooms = list(state.roomDetails)

            # Try to match each expected room with an actual room
            for expected_room in expected_rooms:
                matching_room_index = -1
                for i, actual_room in enumerate(actual_rooms):
                    # Check if this actual room matches the expected room
                    if actual_room.adults != expected_room.get("adults", 0):
                        continue
                    if actual_room.children != expected_room.get("children", 0):
                        continue

                    # Check child ages if specified
                    expected_ages = expected_room.get("child_ages", [])
                    if expected_ages:
                        actual_ages = list(actual_room.child_ages) if actual_room.child_ages else []
                        if sorted(actual_ages) != sorted(expected_ages):
                            continue

                    # Found a match
                    matching_room_index = i
                    break

                if matching_room_index == -1:
                    # No matching room found
                    expected_desc = (
                        f"{expected_room.get('adults', 0)} adults, {expected_room.get('children', 0)} children"
                    )
                    expected_ages = expected_room.get("child_ages", [])
                    if expected_ages:
                        expected_desc += f" (ages: {', '.join(map(str, expected_ages))})"
                    errors.append(f"Expected room configuration not found: {expected_desc}")
                else:
                    # Remove the matched room so it can't be matched again
                    actual_rooms.pop(matching_room_index)

    else:
        # Legacy single-room format - use aggregated fields for backwards compatibility
        if is_specified(params.adults):
            if state.adults != params.adults:
                errors.append(f"Adults count mismatch: expected {params.adults}, got {state.adults}")

        if is_specified(params.children):
            if state.children != params.children:
                errors.append(f"Children count mismatch: expected {params.children}, got {state.children}")

        if is_specified(params.child_ages):
            # Handle None case (not set) vs empty list (set but no children)
            if state.child_ages is None:
                errors.append(f"Child ages not set: expected [{', '.join(map(str, params.child_ages))}], got none")
            elif sorted(state.child_ages) != sorted(params.child_ages):
                errors.append(
                    f"Child ages mismatch: expected [{', '.join(map(str, params.child_ages))}], "
                    f"got [{', '.join(map(str, state.child_ages))}]"
                )

    # Check addFlight if specified
    if is_specified(params.addFlight):
        if state.addFlight != params.addFlight:
            errors.append(f"Add flight mismatch: expected {params.addFlight}, got {state.addFlight}")

    # Check addCar if specified
    if is_specified(params.addCar):
        if state.addCar != params.addCar:
            errors.append(f"Add car mismatch: expected {params.addCar}, got {state.addCar}")

    # Always check if search was submitted
    if not state.isSubmitted:
        errors.append("Search was not submitted")

    # Build actual state summary for debugging
    actual_summary = {
        "destination": state.destination,
        "checkin": state.checkin,
        "checkout": state.checkout,
        "adults": state.adults,
        "children": state.children,
        "child_ages": state.child_ages,
        "rooms": state.rooms,
        "roomDetails": [
            {"adults": room.adults, "children": room.children, "child_ages": room.child_ages}
            for room in state.roomDetails
        ],
        "addFlight": state.addFlight,
        "addCar": state.addCar,
        "isSubmitted": state.isSubmitted,
    }

    return VerificationResult(
        passed=len(errors) == 0, details={"errors": errors, "expected": params.__dict__, "actual": actual_summary}
    )


def create_application_state_from_dict(data: Dict[str, Any]) -> ApplicationState:
    """Create an ApplicationState object from a dictionary (useful for testing)."""
    room_details = []
    for room_data in data.get("roomDetails", []):
        room = RoomDetails(
            adults=room_data.get("adults", 0),
            children=room_data.get("children", 0),
            child_ages=room_data.get("child_ages", []),
        )
        room_details.append(room)

    return ApplicationState(
        destination=data.get("destination", ""),
        checkin=data.get("checkin"),
        checkout=data.get("checkout"),
        adults=data.get("adults", 0),
        children=data.get("children", 0),
        child_ages=data.get("child_ages", []),
        rooms=data.get("rooms", 0),
        roomDetails=room_details,
        addFlight=data.get("addFlight", False),
        addCar=data.get("addCar", False),
        isSubmitted=data.get("isSubmitted", False),
    )


def create_hotel_search_params_from_dict(data: Dict[str, Any]) -> HotelSearchParams:
    """Create a HotelSearchParams object from a dictionary (useful for testing)."""
    return HotelSearchParams(
        city=data.get("city"),
        checkin=data.get("checkin"),
        checkout=data.get("checkout"),
        adults=data.get("adults"),
        children=data.get("children"),
        child_ages=data.get("child_ages"),
        rooms=data.get("rooms"),
        addFlight=data.get("addFlight"),
        addCar=data.get("addCar"),
    )


class InputDict(TypedDict, total=False):
    page: "Page"


class FinalResult(BaseModel):
    """Result model for Travelpedia hotel search task."""

    score: float
    reasoning: str
    current_state: dict[str, Any]
    ground_truth: HotelSearchParams


class HalluminateTravelpediaHotelSearch(BaseMetric):
    """Evaluator that runs verifier for Travelpedia hotel search task."""

    def __init__(self, task: str, values: dict[str, Any], user_metadata: UserMetadata) -> None:
        super().__init__()
        self.task = task
        self.values = values
        self.user_metadata = user_metadata

        self._ground_truth = create_hotel_search_params_from_dict(values)
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
        result = verify_hotel_search_complete(state_obj, self._ground_truth)

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
        return await self._page.evaluate("() => window.__testingAPI.searchForm.getState()")


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


def _format_date_for_task(date_str: str) -> str:
    """Convert YYYY-MM-DD to 'Month Day, Year' format for task statement.

    Example:
        Input:  "2025-12-04"
        Output: "December 4, 2025"
    """
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    return date_obj.strftime("%B %d, %Y")


def _render_values(values: dict[str, Any], timezone: str, timestamp: int) -> dict[str, Any]:
    """Render checkin/checkout date formulas for hotel_search tasks.

    Only render fields that contain date formulas (checkin/checkout).
    All other fields are left unchanged.
    """
    evaluated = dict(values)

    # Only evaluate checkin and checkout if they are formulas
    for key in ("checkin", "checkout"):
        v = evaluated.get(key)
        if isinstance(v, str) and ("now()" in v or "timedelta" in v):
            evaluated[key] = _evaluate_date_formula(v, timezone, timestamp)

    return evaluated


def _render_task(task: str, values: dict[str, Any]) -> str:
    """Replace {checkin}/{checkout} placeholders with formatted dates.

    Example:
        Input task: "Search for a hotel in Tokyo. Set check-in date to {checkin}..."
        Input values: {"checkin": "2025-12-04", "checkout": "2025-12-07"}
        Output: "Search for a hotel in Tokyo. Set check-in date to December 4, 2025..."
    """
    result = task

    for key in ("checkin", "checkout"):
        placeholder = f"{{{key}}}"  # "{checkin}" or "{checkout}"
        v = values.get(key)

        # Only replace if placeholder exists and value is a valid date string (YYYY-MM-DD)
        if placeholder in result and isinstance(v, str) and len(v) == 10:
            result = result.replace(placeholder, _format_date_for_task(v))

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

    # Render the values
    values = _render_values(values, timezone, timestamp)

    # Render the task
    task = _render_task(task, values)

    # Create the eval config
    eval_target = get_import_path(HalluminateTravelpediaHotelSearch)
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
        "task_id": "halluminate_bench/travelpedia/hotel_search/1",
        "task_generation_config_json": json.dumps(
            {
                "_target_": "halluminate_bench.travelpedia.hotel_search.generate_task_config",
                "url": "https://travelpedia.halluminate.ai/?apiKey={HALLUMINATE_API_KEY}",
                "task": (
                    "Search for a hotel in Rio de Janeiro. Set check-in date to {checkin} and check-out to {checkout}. "
                    "Set 1 adult as travelers. Once you have successfully submitted the search, your task is complete "
                    "- do not take any further actions."
                ),
                "values": {
                    "city": "Rio de Janeiro",
                    "checkin": "now() + timedelta(days=291)",
                    "checkout": "now() + timedelta(days=301)",
                    "adults": 1,
                    "children": 0,
                    "child_ages": [],
                },
                "location": "San Francisco, CA, United States",
                "timezone": "America/Los_Angeles",
                "timestamp": None,
            }
        ),
        "env": "sim",
        "domain": "travelpedia",
        "l1_category": "travel",
        "l2_category": "hotel_search",
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
