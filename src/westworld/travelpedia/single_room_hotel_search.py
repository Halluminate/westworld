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
class SingleRoomHotelSearchParams:
    """Expected single room hotel search parameters from the task."""

    city: Optional[str] = None
    checkin: Optional[str] = None
    checkout: Optional[str] = None
    adults: Optional[int] = None
    children: Optional[int] = None
    child_ages: Optional[List[int]] = None


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


def verify_single_room_hotel_search(state: ApplicationState, params: SingleRoomHotelSearchParams) -> VerificationResult:
    """
    Verify that the application state matches the expected single room hotel search parameters.

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

    # Handle single room verification
    if is_specified(params.adults):
        if state.adults != params.adults:
            errors.append(f"Adults count mismatch: expected {params.adults}, got {state.adults}")

    if is_specified(params.children):
        if state.children != params.children:
            errors.append(f"Children count mismatch: expected {params.children}, got {state.children}")

    if is_specified(params.child_ages):
        if not state.child_ages:
            errors.append(f"Child ages not set: expected [{', '.join(map(str, params.child_ages))}], got none")
        elif sorted(state.child_ages) != sorted(params.child_ages):
            errors.append(
                f"Child ages mismatch: expected [{', '.join(map(str, params.child_ages))}], "
                f"got [{', '.join(map(str, state.child_ages))}]"
            )

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


def create_single_room_hotel_search_params_from_dict(data: Dict[str, Any]) -> SingleRoomHotelSearchParams:
    """Create a SingleRoomHotelSearchParams object from a dictionary (useful for testing)."""
    return SingleRoomHotelSearchParams(
        city=data.get("city"),
        checkin=data.get("checkin"),
        checkout=data.get("checkout"),
        adults=data.get("adults"),
        children=data.get("children"),
        child_ages=data.get("child_ages"),
    )


class InputDict(TypedDict, total=False):
    page: "Page"


class FinalResult(BaseModel):
    """Result model for Travelpedia single room hotel search task."""

    score: float
    reasoning: str
    current_state: dict[str, Any]
    ground_truth: SingleRoomHotelSearchParams


class HalluminateTravelpediaSingleRoomHotelSearch(BaseMetric):
    """Evaluator that runs verifier for Travelpedia single room hotel search task."""

    def __init__(self, task: str, values: dict[str, Any], user_metadata: UserMetadata) -> None:
        super().__init__()
        self.task = task
        self.values = values
        self.user_metadata = user_metadata

        self._ground_truth = create_single_room_hotel_search_params_from_dict(values)
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
        result = verify_single_room_hotel_search(state_obj, self._ground_truth)

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


# ============================================================================
# DateUtils Helper Methods (for date calculations)
# ============================================================================


class DateUtils:
    """Helper methods for date calculations."""

    DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    MONTH_NAMES = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    @staticmethod
    def get_first_weekday_after(date: datetime, target_day: int) -> datetime:
        """Get the first occurrence of target_day on or after date."""
        result = date
        while result.weekday() != target_day:
            result += timedelta(days=1)
        return result

    @staticmethod
    def get_last_weekday_in_month(year: int, month: int, target_day: int) -> datetime:
        """Get the last occurrence of target_day in the given month."""
        if month == 12:
            last_day = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            last_day = datetime(year, month + 1, 1) - timedelta(days=1)
        result = last_day
        while result.weekday() != target_day:
            result -= timedelta(days=1)
        return result

    @staticmethod
    def get_nth_weekday_in_month(year: int, month: int, target_day: int, n: int) -> datetime:
        """Get the nth occurrence of target_day in the given month."""
        if target_day < 0 or target_day > 6:
            raise ValueError(f"Invalid target_day: {target_day}. Must be between 0 (Monday) and 6 (Sunday)")
        if n < 1 or n > 5:
            raise ValueError(f"Invalid occurrence: {n}. Must be between 1 and 5")
        date = datetime(year, month, 1)
        count = 0
        while date.month == month:
            if date.weekday() == target_day:
                count += 1
                if count == n:
                    return date
            date += timedelta(days=1)
        raise ValueError(f"No {n}th occurrence of weekday {target_day} found in {year}/{month}")

    @staticmethod
    def add_months(date: datetime, months: int) -> datetime:
        """Add months to a date, handling edge cases."""
        year = date.year
        month = date.month + months
        while month > 12:
            year += 1
            month -= 12
        while month < 1:
            year -= 1
            month += 12
        try:
            return datetime(year, month, date.day)
        except ValueError:
            last_day = (datetime(year, month + 1, 1) - timedelta(days=1)).day
            return datetime(year, month, min(date.day, last_day))

    @staticmethod
    def get_first_monday_of_month(year: int, month: int) -> datetime:
        """Get the first Monday of a given month."""
        return DateUtils.get_nth_weekday_in_month(year, month, 0, 1)

    @staticmethod
    def get_last_monday_of_month(year: int, month: int) -> datetime:
        """Get the last Monday of a given month."""
        last_day = datetime(year, month + 1, 1) - timedelta(days=1)
        while last_day.weekday() != 0:
            last_day -= timedelta(days=1)
        return last_day

    @staticmethod
    def get_nth_monday_of_month(year: int, month: int, n: int) -> datetime:
        """Get the nth Monday of a given month."""
        return DateUtils.get_nth_weekday_in_month(year, month, 0, n)

    @staticmethod
    def get_nth_thursday_of_month(year: int, month: int, n: int) -> datetime:
        """Get the nth Thursday of a given month."""
        return DateUtils.get_nth_weekday_in_month(year, month, 3, n)


def _evaluate_date_formula(formula_str: str, timezone: str, timestamp: int) -> str:
    """Safely evaluate a date formula string and return YYYY-MM-DD string.

    Example:
        Input:  formula_str="now() + timedelta(days=45)", timezone="America/Los_Angeles", timestamp=1715702400
        Output: "2024-05-15"

        Input:  formula_str="now() + timedelta(days=45)", timezone="Asia/Tokyo", timestamp=1715702400
        Output: "2024-05-16"
    """

    def now():
        return datetime.fromtimestamp(timestamp, tz=ZoneInfo(timezone))

    safe_globals = {"timedelta": timedelta, "now": now}
    result = eval(formula_str, safe_globals)
    return result.strftime("%Y-%m-%d")


def _render_values(values: dict[str, Any], timezone: str, timestamp: int) -> dict[str, Any]:
    """Render date formulas and calculate dates based on pattern metadata for single-room-hotel-search tasks.

    This function:
    1. Evaluates the base date formula to get a concrete datetime
    2. Based on pattern type and parameters, calculates the actual checkin/checkout dates
    3. Returns evaluated values with concrete dates
    """
    evaluated = dict(values)

    # Get base formula and pattern metadata
    base_formula = values.get("_baseDateFormula")
    pattern_metadata = values.get("_patternMetadata")

    if not base_formula or not pattern_metadata:
        # Fallback: just evaluate checkin/checkout formulas directly
        for key in ("checkin", "checkout"):
            v = evaluated.get(key)
            if isinstance(v, str) and ("now()" in v or "timedelta" in v):
                evaluated[key] = _evaluate_date_formula(v, timezone, timestamp)
        return evaluated

    # Evaluate base formula to get concrete datetime
    base_date = _evaluate_date_formula(base_formula, timezone, timestamp)
    base_datetime = datetime.strptime(base_date, "%Y-%m-%d")

    pattern_type = pattern_metadata.get("patternType")
    pattern_params = pattern_metadata.get("patternParams", {})

    # Calculate checkin and checkout based on pattern type
    checkin_dt = None
    checkout_dt = None

    if pattern_type == "first_weekend_after":
        # First Saturday strictly after base date (not on or after)
        target_weekday = pattern_params.get("targetWeekday", 5)  # Saturday = 5
        # Start from the day after base_date to ensure we get the weekend AFTER, not including
        next_day = base_datetime + timedelta(days=1)
        saturday = DateUtils.get_first_weekday_after(next_day, target_weekday)
        checkin_dt = saturday
        checkout_dt = saturday + timedelta(days=1)  # Sunday

    elif pattern_type == "first_day_of_month":
        months_ahead = pattern_params.get("monthsAhead", 2)
        day_index = pattern_params.get("dayIndex", 0)
        nights = pattern_params.get("nights", 2)

        target_date = DateUtils.add_months(base_datetime, months_ahead)
        target_date = target_date.replace(day=1)
        first_occurrence = DateUtils.get_nth_weekday_in_month(target_date.year, target_date.month, day_index, 1)
        checkin_dt = first_occurrence
        checkout_dt = first_occurrence + timedelta(days=nights)

    elif pattern_type == "last_week_of_month":
        months_ahead = pattern_params.get("monthsAhead", 1)
        target_date = DateUtils.add_months(base_datetime, months_ahead)

        # Get last Sunday of month
        last_sunday = DateUtils.get_last_weekday_in_month(target_date.year, target_date.month, 6)

        # Ensure full week (Sunday to Saturday)
        last_day_of_month = (
            datetime(target_date.year, target_date.month + 1, 1) - timedelta(days=1)
            if target_date.month < 12
            else datetime(target_date.year + 1, 1, 1) - timedelta(days=1)
        )
        if last_sunday.day + 6 > last_day_of_month.day:
            last_sunday = last_sunday - timedelta(days=7)

        checkin_dt = last_sunday
        checkout_dt = last_sunday + timedelta(days=6)  # Saturday

    elif pattern_type == "duration_based":
        duration_days = pattern_params.get("durationDays", 3)
        checkin_dt = base_datetime
        checkout_dt = base_datetime + timedelta(days=duration_days)

    elif pattern_type == "midweek":
        weeks_ahead = pattern_params.get("weeksAhead", 4)
        start_weekday = pattern_params.get("startWeekday", 1)  # Tuesday
        _end_weekday = pattern_params.get("endWeekday", 3)  # Thursday

        target_date = base_datetime + timedelta(days=weeks_ahead * 7)
        tuesday = DateUtils.get_first_weekday_after(target_date, start_weekday)
        checkin_dt = tuesday
        checkout_dt = tuesday + timedelta(days=2)  # Thursday

    elif pattern_type == "long_weekend":
        weeks_ahead = pattern_params.get("weeksAhead", 4)
        target_date = base_datetime + timedelta(days=weeks_ahead * 7)
        thursday = DateUtils.get_first_weekday_after(target_date, 3)  # Thursday = 3
        checkin_dt = thursday
        checkout_dt = thursday + timedelta(days=4)  # Monday

    elif pattern_type == "weeks_after_ordinal":
        months_ahead = pattern_params.get("monthsAhead", 3)
        day_index = pattern_params.get("dayIndex", 0)
        ordinal_index = pattern_params.get("ordinalIndex", 0)
        weeks_after = pattern_params.get("weeksAfter", 1)
        nights = pattern_params.get("nights", 3)

        target_date = DateUtils.add_months(base_datetime, months_ahead)
        target_date = target_date.replace(day=1)
        nth_day = DateUtils.get_nth_weekday_in_month(target_date.year, target_date.month, day_index, ordinal_index + 1)
        checkin_dt = nth_day + timedelta(days=weeks_after * 7)
        checkout_dt = checkin_dt + timedelta(days=nights)

    elif pattern_type == "between_dates":
        start_days_ahead = pattern_params.get("startDaysAhead", 30)
        _end_days_ahead = pattern_params.get("endDaysAhead", 60)
        random_days_from_start = pattern_params.get("randomDaysFromStart", 2)
        nights = pattern_params.get("nights", 3)

        start_boundary = base_datetime + timedelta(days=start_days_ahead)
        checkin_dt = start_boundary + timedelta(days=random_days_from_start)
        checkout_dt = checkin_dt + timedelta(days=nights)

    else:
        # Fallback: just evaluate formulas directly
        for key in ("checkin", "checkout"):
            v = evaluated.get(key)
            if isinstance(v, str) and ("now()" in v or "timedelta" in v):
                evaluated[key] = _evaluate_date_formula(v, timezone, timestamp)
        return evaluated

    # Convert datetime objects to YYYY-MM-DD strings
    if checkin_dt:
        evaluated["checkin"] = checkin_dt.strftime("%Y-%m-%d")
    if checkout_dt:
        evaluated["checkout"] = checkout_dt.strftime("%Y-%m-%d")

    return evaluated


def _render_task(task: str, values: dict[str, Any]) -> str:
    """Replace placeholders in task statement with formatted date information.

    Placeholders can include:
    - {month}, {day}, {year} - from base date
    - {ordinal} - ordinal description (first, second, etc.)
    - {thursdayMonth}, {thursdayDay}, {mondayMonth}, {mondayDay} - for long weekend
    - {startMonth}, {startDay}, {endMonth}, {endDay} - for between dates
    """
    result = task

    # For "first_weekend_after" pattern, use base date for {month} and {day}
    # For other patterns, use checkin date
    pattern_metadata = values.get("_patternMetadata")
    use_base_date = pattern_metadata and pattern_metadata.get("patternType") == "first_weekend_after"

    if use_base_date:
        # Use base date for {month} and {day} placeholders
        base_formula = values.get("_baseDateFormula")
        if base_formula:
            try:
                # Note: base_formula should already be evaluated by this point
                # But if it contains formula syntax, we need timestamp
                # For simplicity, we'll assume values already contain evaluated dates
                checkin_str = values.get("checkin")
                if checkin_str:
                    base_dt = datetime.strptime(checkin_str, "%Y-%m-%d") - timedelta(days=1)
                    month_name = DateUtils.MONTH_NAMES[base_dt.month - 1]
                    day = base_dt.day
                    year = base_dt.year

                    # Replace basic placeholders with base date
                    result = result.replace("{month}", month_name)
                    result = result.replace("{day}", str(day))
                    result = result.replace("{year}", str(year))
            except (ValueError, KeyError, IndexError):
                pass
    else:
        # Get checkin date for extracting month/day/year (original logic)
        checkin_str = values.get("checkin")
        if not checkin_str:
            return result

        try:
            checkin_dt = datetime.strptime(checkin_str, "%Y-%m-%d")
            month_name = DateUtils.MONTH_NAMES[checkin_dt.month - 1]
            day = checkin_dt.day
            year = checkin_dt.year

            # Replace basic placeholders
            result = result.replace("{month}", month_name)
            result = result.replace("{day}", str(day))
            result = result.replace("{year}", str(year))
        except (ValueError, KeyError, IndexError):
            pass

    # Handle ordinal calculation (which week of month) - use checkin date
    if "{ordinal}" in result:
        checkin_str = values.get("checkin")
        if checkin_str:
            try:
                checkin_dt = datetime.strptime(checkin_str, "%Y-%m-%d")
                week_of_month = (checkin_dt.day - 1) // 7 + 1
                ordinals = ["first", "second", "third", "fourth", "fifth"]
                ordinal = ordinals[week_of_month - 1] if week_of_month <= 5 else "last"
                result = result.replace("{ordinal}", ordinal)
            except (ValueError, KeyError, IndexError):
                pass

    # Handle long weekend placeholders
    if "{thursdayMonth}" in result or "{thursdayDay}" in result:
        checkin_str = values.get("checkin")
        if checkin_str:
            try:
                # Checkin is Thursday for long weekend
                thursday_dt = datetime.strptime(checkin_str, "%Y-%m-%d")
                thursday_month = DateUtils.MONTH_NAMES[thursday_dt.month - 1]
                result = result.replace("{thursdayMonth}", thursday_month)
                result = result.replace("{thursdayDay}", str(thursday_dt.day))
            except (ValueError, KeyError, IndexError):
                pass

    if "{mondayMonth}" in result or "{mondayDay}" in result:
        # Checkout is Monday for long weekend
        checkout_str = values.get("checkout")
        if checkout_str:
            try:
                monday_dt = datetime.strptime(checkout_str, "%Y-%m-%d")
                monday_month = DateUtils.MONTH_NAMES[monday_dt.month - 1]
                result = result.replace("{mondayMonth}", monday_month)
                result = result.replace("{mondayDay}", str(monday_dt.day))
            except (ValueError, KeyError, IndexError):
                pass

    # Handle between dates placeholders
    if "{startMonth}" in result or "{startDay}" in result:
        checkin_str = values.get("checkin")
        if checkin_str:
            try:
                # For between dates, checkin is the start
                checkin_dt = datetime.strptime(checkin_str, "%Y-%m-%d")
                start_month = DateUtils.MONTH_NAMES[checkin_dt.month - 1]
                result = result.replace("{startMonth}", start_month)
                result = result.replace("{startDay}", str(checkin_dt.day))
            except (ValueError, KeyError, IndexError):
                pass

    if "{endMonth}" in result or "{endDay}" in result:
        # For between dates, we need to calculate end boundary
        # This is approximate - we'll use checkout as end
        checkout_str = values.get("checkout")
        if checkout_str:
            try:
                end_dt = datetime.strptime(checkout_str, "%Y-%m-%d")
                end_month = DateUtils.MONTH_NAMES[end_dt.month - 1]
                result = result.replace("{endMonth}", end_month)
                result = result.replace("{endDay}", str(end_dt.day))
            except (ValueError, KeyError, IndexError):
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

    # Render the values
    values = _render_values(values, timezone, timestamp)

    # Render the task
    task = _render_task(task, values)

    # Create the eval config
    eval_target = get_import_path(HalluminateTravelpediaSingleRoomHotelSearch)
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
        "task_id": "halluminate_bench/travelpedia/single_room_hotel_search/1",
        "task_generation_config_json": json.dumps(
            {
                "_target_": "halluminate_bench.travelpedia.single_room_hotel_search.generate_task_config",
                "url": "https://travelpedia.halluminate.ai/?apiKey={HALLUMINATE_API_KEY}",
                "task": (
                    "Book a four-day getaway starting from {month} {day}, {year} in Tel Aviv. "
                    "The booking is for 2 adults. Once you have successfully submitted the search, "
                    "your task is complete - do not take any further actions."
                ),
                "values": {
                    "city": "Tel Aviv",
                    "checkin": "now() + timedelta(days=161)",
                    "checkout": "now() + timedelta(days=165)",
                    "adults": 2,
                    "children": 0,
                    "_baseDateFormula": "now() + timedelta(days=161)",
                    "_patternMetadata": {
                        "patternType": "duration_based",
                        "patternParams": {"durationDays": 4, "durationDescription": "four-day getaway"},
                        "baseFormula": "now() + timedelta(days=161)",
                    },
                },
                "location": "San Francisco, CA, United States",
                "timezone": "America/Los_Angeles",
                "timestamp": None,
            }
        ),
        "env": "sim",
        "domain": "travelpedia",
        "l1_category": "travel",
        "l2_category": "single_room_hotel_search",
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
