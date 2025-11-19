import json
import os
from dataclasses import dataclass, field
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
# Core domain models
# =========================


@dataclass
class TravelClass:
    value: str
    label: str


@dataclass
class Airport:
    name: str
    city: str
    country: str
    iata: str
    lat: float
    lon: float
    tz: str
    routeCount: int
    airlines: Dict[str, float]
    hubScore: int


@dataclass
class Airline:
    code: str
    name: str
    country: str
    route_count: int
    importance: float
    hubs: List[dict]


TRAVEL_CLASS_OPTIONS = [
    TravelClass(value="economy", label="Economy"),
    TravelClass(value="premium-economy", label="Premium economy"),
    TravelClass(value="business", label="Business"),
    TravelClass(value="first", label="First"),
]


@dataclass
class TaskData:
    origin: Airport
    destination: Airport
    range_one: tuple[datetime, datetime]
    range_two: tuple[datetime, datetime]
    range_three: tuple[datetime, datetime]
    travel_class: TravelClass
    passengers: int
    airline: Airline


# =========================
# Flight API models
# =========================


@dataclass
class CabinPrice:
    cabin: str
    priceUsd: int
    seatsRemaining: int


@dataclass
class FlightSegment:
    flightNumber: str
    airline: str
    airlineName: str
    aircraft: str
    depart: str
    arrive: str
    durationMin: int
    origin: str
    destination: str


@dataclass
class Flight:
    flightNumber: str
    airline: str
    airlineName: str
    aircraft: str
    depart: Optional[str] = None
    arrive: Optional[str] = None
    durationMin: Optional[int] = None
    distanceKm: Optional[int] = None
    cabinPricing: List[CabinPrice] = field(default_factory=list)
    emissions: Optional[int] = None
    priceUsd: Optional[int] = None
    originAirport: Optional[str] = None
    destinationAirport: Optional[str] = None
    segments: List[FlightSegment] = field(default_factory=list)
    totalDurationMin: Optional[int] = None
    layoverDurationMin: Optional[int] = None
    layoverAirport: Optional[str] = None


@dataclass
class FlightResults:
    flights: List[Flight]
    expandedOrigins: Optional[List[str]] = None
    expandedDestinations: Optional[List[str]] = None


# =========================
# API helpers
# =========================


def fetch_flights(
    origin: str,
    destination: str,
    departure_date: str,
    return_date: str,
    base_url: Optional[str],
    travel_class: Optional[str] = None,
    adults: int = 1,
    trip_type: str = "round-trip",
) -> Dict[str, Any]:
    provided_base = base_url or os.getenv("FLIGHTS_BASE_URL") or "https://flights.halluminate.ai"
    base = provided_base.rstrip("/")
    if base.endswith("/api/flights"):
        flights_endpoint = base
    else:
        flights_endpoint = f"{base}/api/flights"
    params = {
        "origin": origin,
        "destination": destination,
        "departureDate": departure_date,
        "returnDate": return_date,
        "tripType": trip_type,
        "travelClass": (travel_class or "economy"),
        "adults": adults,
    }
    try:
        response = requests.get(f"{flights_endpoint}?{urlencode(params)}", timeout=30)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to query flights API at {flights_endpoint}: {e}") from e
    except ValueError as e:
        raise RuntimeError(f"Invalid JSON from flights API at {flights_endpoint}: {e}") from e


# =========================
# Verification helpers
# =========================


def _normalize_cabin_name(travel_class_value: str) -> str:
    mapping = {
        "economy": "ECONOMY",
        "premium-economy": "PREMIUM_ECONOMY",
        "business": "BUSINESS",
        "first": "FIRST",
    }
    return mapping.get(travel_class_value.lower(), travel_class_value.upper())


def _effective_price_usd(flight: Flight, cabin_name: str) -> Optional[int]:
    # Prefer price for the requested cabin if available; otherwise fall back to overall price
    for cp in flight.cabinPricing:
        if cp.cabin == cabin_name:
            return cp.priceUsd
    if flight.priceUsd is not None:
        return flight.priceUsd
    return None


def _find_cabin_price_usd(flight: Flight, cabin_name: str) -> Optional[int]:
    """Return the priceUsd for the exact requested cabin, if available."""
    for cp in flight.cabinPricing:
        if cp.cabin == cabin_name:
            return cp.priceUsd
    return None


def _departure_arrival_times(flight: Flight) -> tuple[Optional[str], Optional[str]]:
    """Return (depart, arrive) timestamps with timezone when possible.

    Falls back to first/last segment timestamps if top-level fields are missing.
    """
    if flight.depart and flight.arrive:
        return flight.depart, flight.arrive
    if flight.segments:
        depart_time = flight.segments[0].depart
        arrive_time = flight.segments[-1].arrive
        return depart_time, arrive_time
    return None, None


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        v = value
        if v.endswith("Z"):
            v = v.replace("Z", "+00:00")
        return datetime.fromisoformat(v)
    except Exception:
        try:
            return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None


def _format_clock_time(dt: Optional[datetime]) -> Optional[str]:
    if not dt:
        return None
    # Use 12-hour without leading zero, with AM/PM
    t = dt.strftime("%I:%M %p")
    if t.startswith("0"):
        t = t[1:]
    return t


def _format_time_range(depart: Optional[str], arrive: Optional[str]) -> Optional[str]:
    dep_dt = _parse_iso_datetime(depart)
    arr_dt = _parse_iso_datetime(arrive)
    if not dep_dt or not arr_dt:
        return None
    dep_str = _format_clock_time(dep_dt)
    arr_str = _format_clock_time(arr_dt)
    if not dep_str or not arr_str:
        return None
    return f"{dep_str} – {arr_str}"


def _flight_duration_minutes(flight: Flight, depart: Optional[str], arrive: Optional[str]) -> Optional[int]:
    if getattr(flight, "totalDurationMin", None) is not None:
        return flight.totalDurationMin  # type: ignore[return-value]
    if getattr(flight, "durationMin", None) is not None:
        return flight.durationMin  # type: ignore[return-value]
    dep_dt = _parse_iso_datetime(depart)
    arr_dt = _parse_iso_datetime(arrive)
    if dep_dt and arr_dt:
        delta = arr_dt - dep_dt
        minutes = int(delta.total_seconds() // 60)
        return minutes if minutes >= 0 else None
    return None


def _format_duration(minutes: Optional[int]) -> Optional[str]:
    if minutes is None:
        return None
    hours = minutes // 60
    rem = minutes % 60
    if hours and rem:
        return f"{hours} hr {rem} min"
    if hours:
        return f"{hours} hr"
    return f"{rem} min"


def _format_price_usd(amount: Optional[int]) -> Optional[str]:
    if amount is None:
        return None
    return f"${amount:,}"


def _origin_destination_airports(flight: Flight) -> tuple[Optional[str], Optional[str]]:
    """Return (originAirport, destinationAirport) codes.

    Falls back to first/last segment airports if top-level fields are missing.
    """
    if flight.originAirport and flight.destinationAirport:
        return flight.originAirport, flight.destinationAirport
    if flight.segments:
        origin_code = flight.segments[0].origin
        destination_code = flight.segments[-1].destination
        return origin_code, destination_code
    return None, None


def _build_expected_flight(
    flight: Flight,
    num_passengers: int,
    cabin_normalized: str,
) -> Dict[str, Any]:
    depart_time, arrive_time = _departure_arrival_times(flight)
    origin_code, destination_code = _origin_destination_airports(flight)
    cabin_price = _find_cabin_price_usd(flight, cabin_normalized)
    per_passenger_price = cabin_price if cabin_price is not None else _effective_price_usd(flight, cabin_normalized)
    total_price = (per_passenger_price * num_passengers) if isinstance(per_passenger_price, int) else None
    formatted_price = _format_price_usd(total_price) if isinstance(total_price, int) else None
    formatted_time_range = _format_time_range(depart_time, arrive_time)
    duration_minutes = _flight_duration_minutes(flight, depart_time, arrive_time)
    formatted_duration = _format_duration(duration_minutes) if duration_minutes is not None else None
    return {
        "flight_number": flight.flightNumber,
        "airline_name": flight.airlineName,
        "departure_time": depart_time,
        "arrival_time": arrive_time,
        "source_airport_code": origin_code,
        "destination_airport_code": destination_code,
        "total_price": total_price,
        "formatted_total_price": formatted_price,
        "formatted_time_range": formatted_time_range,
        "duration_minutes": duration_minutes,
        "formatted_duration": formatted_duration,
    }


def filter_flights(
    flights: List[Flight],
    class_name: str,
    num_passengers: int,
    include_airline: Optional[str] = None,
) -> List[Flight]:
    """
    Filter flights for direct service with requested cabin and seats available.

    Returns only the cheapest flight (simplified version).

    Args:
        include_airline: Ignored in simplified version (kept for compatibility)
    """
    cabin = _normalize_cabin_name(class_name)

    def has_cabin_and_seats(f: Flight) -> bool:
        for cp in f.cabinPricing:
            if cp.cabin == cabin and cp.seatsRemaining >= num_passengers:
                return True
        return False

    direct_flights = [f for f in flights if not f.segments]
    eligible = [f for f in direct_flights if has_cabin_and_seats(f)]

    if not eligible:
        return []

    def price_key(f: Flight) -> int:
        p = _effective_price_usd(f, cabin)
        return p if p is not None else 10**9

    eligible.sort(key=price_key)

    # Return only cheapest (simplified version)
    return [eligible[0]]


def create_task_data_from_dict(data: Dict[str, Any]) -> TaskData:
    """Create a TaskData object from a dictionary."""

    def parse_airport(airport_data: Dict[str, Any]) -> Airport:
        return Airport(
            name=airport_data["name"],
            city=airport_data["city"],
            country=airport_data["country"],
            iata=airport_data["iata"],
            lat=float(airport_data["lat"]),
            lon=float(airport_data["lon"]),
            tz=airport_data["tz"],
            routeCount=int(airport_data["routeCount"]),
            airlines=dict(airport_data.get("airlines", {})),
            hubScore=int(airport_data["hubScore"]),
        )

    def parse_airline(airline_data: Dict[str, Any]) -> Airline:
        return Airline(
            code=airline_data["code"],
            name=airline_data["name"],
            country=airline_data["country"],
            route_count=int(airline_data["route_count"]),
            importance=float(airline_data["importance"]),
            hubs=airline_data.get("hubs", []),
        )

    def parse_range(r: Any) -> tuple[datetime, datetime]:
        if isinstance(r, list) and len(r) == 2:
            start, end = r
            return (_parse_date(start), _parse_date(end))
        raise ValueError(f"Invalid range format: {r}")

    def _parse_date(value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            # Try ISO format first
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                # Fallback to common patterns
                for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                    try:
                        return datetime.strptime(value, fmt)
                    except ValueError:
                        continue
        raise ValueError(f"Unsupported date value: {value!r}")

    origin = parse_airport(data["origin"])
    destination = parse_airport(data["destination"])
    range_one = parse_range(data["range_one"])
    range_two = parse_range(data["range_two"])
    range_three = parse_range(data["range_three"])
    travel_class = TravelClass(**data["travel_class"])
    passengers = int(data["passengers"])
    airline = parse_airline(data["airline"])

    return TaskData(
        origin=origin,
        destination=destination,
        range_one=range_one,
        range_two=range_two,
        range_three=range_three,
        travel_class=travel_class,
        passengers=passengers,
        airline=airline,
    )


def verify_date_range_search_simplified_complete(
    current_state: Dict[str, Any], task_data: TaskData, agent_response: Optional[str] = None
) -> Dict[str, Any]:
    """Verify that the agent responded with flight information for all three date ranges.

    Simplified version: Only checks for cheapest flights (not airline-specific flights).

    Steps:
    - Fetch flights for all three date ranges
    - Filter to direct flights with requested cabin and enough seats
    - Check that expected data (flight numbers, prices, times, durations) appear in agent response
    """

    date_ranges = [task_data.range_one, task_data.range_two, task_data.range_three]

    origin = task_data.origin.iata
    destination = task_data.destination.iata
    class_value = task_data.travel_class.value
    num_passengers = task_data.passengers

    results: dict[str, dict[str, list[Flight]]] = {
        "date_range_one": {"departure_flights": [], "return_flights": []},
        "date_range_two": {"departure_flights": [], "return_flights": []},
        "date_range_three": {"departure_flights": [], "return_flights": []},
    }

    range_keys = ["date_range_one", "date_range_two", "date_range_three"]

    for idx, (dep, ret) in enumerate(date_ranges):
        departure_date = dep.strftime("%Y-%m-%d")
        return_date = ret.strftime("%Y-%m-%d")
        api_response = fetch_flights(
            origin=origin,
            destination=destination,
            departure_date=departure_date,
            return_date=return_date,
            base_url=None,
            travel_class=class_value,
            adults=num_passengers,
        )

        departing_flights = [
            Flight(
                flightNumber=f.get("flightNumber", ""),
                airline=f.get("airline", ""),
                airlineName=f.get("airlineName", ""),
                aircraft=f.get("aircraft", ""),
                depart=f.get("depart"),
                arrive=f.get("arrive"),
                durationMin=f.get("durationMin"),
                distanceKm=f.get("distanceKm"),
                cabinPricing=[
                    CabinPrice(cabin=cp["cabin"], priceUsd=cp["priceUsd"], seatsRemaining=cp["seatsRemaining"])
                    for cp in f.get("cabinPricing", [])
                ],
                emissions=f.get("emissions"),
                priceUsd=f.get("priceUsd"),
                originAirport=f.get("originAirport"),
                destinationAirport=f.get("destinationAirport"),
                segments=[
                    FlightSegment(
                        flightNumber=s.get("flightNumber", ""),
                        airline=s.get("airline", ""),
                        airlineName=s.get("airlineName", ""),
                        aircraft=s.get("aircraft", ""),
                        depart=s.get("depart", ""),
                        arrive=s.get("arrive", ""),
                        durationMin=s.get("durationMin", 0),
                        origin=s.get("origin", ""),
                        destination=s.get("destination", ""),
                    )
                    for s in f.get("segments", [])
                ],
                totalDurationMin=f.get("totalDurationMin"),
                layoverDurationMin=f.get("layoverDurationMin"),
                layoverAirport=f.get("layoverAirport"),
            )
            for f in api_response.get("departingFlights", [])
        ]

        returning_flights = [
            Flight(
                flightNumber=f.get("flightNumber", ""),
                airline=f.get("airline", ""),
                airlineName=f.get("airlineName", ""),
                aircraft=f.get("aircraft", ""),
                depart=f.get("depart"),
                arrive=f.get("arrive"),
                durationMin=f.get("durationMin"),
                distanceKm=f.get("distanceKm"),
                cabinPricing=[
                    CabinPrice(cabin=cp["cabin"], priceUsd=cp["priceUsd"], seatsRemaining=cp["seatsRemaining"])
                    for cp in f.get("cabinPricing", [])
                ],
                emissions=f.get("emissions"),
                priceUsd=f.get("priceUsd"),
                originAirport=f.get("originAirport"),
                destinationAirport=f.get("destinationAirport"),
                segments=[
                    FlightSegment(
                        flightNumber=s.get("flightNumber", ""),
                        airline=s.get("airline", ""),
                        airlineName=s.get("airlineName", ""),
                        aircraft=s.get("aircraft", ""),
                        depart=s.get("depart", ""),
                        arrive=s.get("arrive", ""),
                        durationMin=s.get("durationMin", 0),
                        origin=s.get("origin", ""),
                        destination=s.get("destination", ""),
                    )
                    for s in f.get("segments", [])
                ],
                totalDurationMin=f.get("totalDurationMin"),
                layoverDurationMin=f.get("layoverDurationMin"),
                layoverAirport=f.get("layoverAirport"),
            )
            for f in api_response.get("returningFlights", [])
        ]

        filtered_departing = filter_flights(departing_flights, class_value, num_passengers)
        filtered_returning = filter_flights(returning_flights, class_value, num_passengers)

        key = range_keys[idx]
        results[key]["departure_flights"] = filtered_departing
        results[key]["return_flights"] = filtered_returning

    # Build structured expected flights per date range
    cabin = _normalize_cabin_name(class_value)
    expected_flights: Dict[str, Dict[str, Any]] = {}
    for key, v in results.items():
        dep_flights = v["departure_flights"]
        ret_flights = v["return_flights"]

        cheapest_dep = dep_flights[0] if dep_flights else None
        cheapest_ret = ret_flights[0] if ret_flights else None

        expected_flights[key] = {
            "cheapest_departure_flight": (
                _build_expected_flight(cheapest_dep, num_passengers, cabin) if cheapest_dep else None
            ),
            "cheapest_return_flight": (
                _build_expected_flight(cheapest_ret, num_passengers, cabin) if cheapest_ret else None
            ),
        }

    text_lower = (agent_response or "").lower()
    verifications: List[Dict[str, Any]] = []
    for key, group in expected_flights.items():
        for bucket in ["cheapest_departure_flight", "cheapest_return_flight"]:
            ef = group.get(bucket)
            if not ef:
                continue
            # Build candidate strings for matching: flight number, total price, time range, duration
            candidates: Dict[str, Optional[str]] = {
                "flight_number": str(ef.get("flight_number") or ""),
                "price": ef.get("formatted_total_price"),
                "time_range": ef.get("formatted_time_range"),
                "duration": ef.get("formatted_duration"),
            }
            # Also support a variant with hyphen-minus instead of en-dash for time range
            time_range_val = candidates.get("time_range")
            if time_range_val and " – " in time_range_val:
                candidates["time_range_alt"] = time_range_val.replace(" – ", " - ")
            matched_by: List[str] = []
            for name, value in candidates.items():
                if not value:
                    continue
                if str(value).lower() in text_lower:
                    matched_by.append(name)
            verifications.append(
                {
                    "bucket": f"{key}.{bucket}",
                    "matched": bool(matched_by),
                    "matched_by": matched_by,
                    "candidates": {k: v for k, v in candidates.items() if v},
                }
            )

    success = all(v.get("matched", False) for v in verifications)

    return {
        "success": success,
        "verifications": verifications,
        "expected_flights": expected_flights,
        "observed_state": current_state,
    }


# =========================
# Evaluator
# =========================


class InputDict(TypedDict, total=False):
    page: "Page"
    answer_message: str


class FinalResult(BaseModel):
    """Result model for Noodle flights date range search simplified task."""

    score: float
    reasoning: str
    current_state: dict[str, Any]
    ground_truth: TaskData


class HalluminateNoodleFlightsDateRangeSearchSimplified(BaseMetric):
    """Evaluator that runs verifier for Noodle flights date range search simplified task."""

    def __init__(self, task: str, values: dict[str, Any], user_metadata: UserMetadata) -> None:
        super().__init__()
        self.task = task
        self.values = values
        self.user_metadata = user_metadata

        self._ground_truth = create_task_data_from_dict(values)
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

        result = verify_date_range_search_simplified_complete(current_state, self._ground_truth, self._answer_message)

        if result["success"]:
            return FinalResult(
                score=1.0,
                reasoning="Passed all checks",
                current_state=current_state,
                ground_truth=self._ground_truth,
            )
        else:
            reasoning = (
                "Failed (verification): "
                + "; ".join([str(i) for i in result["verifications"]])
                + "; expected: "
                + str(result["expected_flights"])
            )
            return FinalResult(
                score=0.0,
                reasoning=reasoning,
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


def _format_date_range(start_date: datetime, end_date: datetime) -> str:
    """Format date range for task statement.

    Example:
        Input:  datetime(2026, 6, 27), datetime(2026, 6, 30)
        Output: "June 27-30, 2026"
    """
    if start_date.month == end_date.month and start_date.year == end_date.year:
        return f"{start_date.strftime('%B')} {start_date.day}-{end_date.day}, {start_date.year}"
    if start_date.year == end_date.year:
        return (
            f"{start_date.strftime('%B')} {start_date.day}-{end_date.strftime('%B')} {end_date.day}, {start_date.year}"
        )
    return f"{start_date.strftime('%B')} {start_date.day}-{end_date.strftime('%B')} {end_date.day}, {end_date.year}"


def _render_values(values: dict[str, Any], timezone: str, timestamp: int) -> dict[str, Any]:
    """Render date range formulas for date_range_search_simplified tasks.

    Only render fields that contain date formulas (range_one, range_two, range_three).
    All other fields are left unchanged.
    """
    evaluated = dict(values)

    # Evaluate range formulas if present
    for range_key in ["range_one", "range_two", "range_three"]:
        range_val = evaluated.get(range_key)
        if isinstance(range_val, list) and len(range_val) == 2:
            dep_formula = range_val[0]
            ret_formula = range_val[1]

            # Check if they are formulas (strings containing "now()" or "timedelta")
            if isinstance(dep_formula, str) and ("now()" in dep_formula or "timedelta" in dep_formula):
                dep_dt = _evaluate_date_formula(dep_formula, timezone, timestamp)
                ret_dt = _evaluate_date_formula(ret_formula, timezone, timestamp)
                # Store as ISO format strings
                evaluated[range_key] = [dep_dt.isoformat(), ret_dt.isoformat()]

    return evaluated


def _render_task(task: str, values: dict[str, Any]) -> str:
    """Replace {rangeOne}, {rangeTwo}, {rangeThree} placeholders with formatted date ranges.

    Example:
        Input task: "Find flights for {rangeOne}, {rangeTwo}, and {rangeThree}..."
        Input values: {"range_one": ["2025-12-04T00:00:00", "2025-12-07T00:00:00"], ...}
        Output: "Find flights for December 4-7, 2025, December 11-14, 2025, and December 18-21, 2025..."
    """
    result = task

    # Replace placeholders for each range
    for placeholder, range_key in [
        ("{rangeOne}", "range_one"),
        ("{rangeTwo}", "range_two"),
        ("{rangeThree}", "range_three"),
    ]:
        if placeholder in result:
            range_val = values.get(range_key)
            if isinstance(range_val, list) and len(range_val) == 2:
                dep_str, ret_str = range_val
                # Parse ISO format strings to datetime objects for formatting
                try:
                    dep_dt = datetime.fromisoformat(dep_str.replace("Z", "+00:00"))
                    ret_dt = datetime.fromisoformat(ret_str.replace("Z", "+00:00"))
                    date_range_str = _format_date_range(dep_dt, ret_dt)
                    result = result.replace(placeholder, date_range_str)
                except (ValueError, AttributeError):
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

    # Render the values
    values = _render_values(values, timezone, timestamp)

    # Render the task
    task = _render_task(task, values)

    # Create the eval config
    eval_target = get_import_path(HalluminateNoodleFlightsDateRangeSearchSimplified)
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
        "task_id": "halluminate_bench/noodle_flights/date_range_search_simplified/1",
        "task_generation_config_json": json.dumps(
            {
                "_target_": "halluminate_bench.noodle_flights.date_range_search_simplified.generate_task_config",
                "url": "https://flights.halluminate.ai/?apiKey={HALLUMINATE_API_KEY}",
                "task": (
                    "Search for round-trip direct Premium economy class flights from BKK to VIE for 2 adult passengers."
                    " I want to see flight options for {rangeOne}, {rangeTwo}, and {rangeThree}. Extract exact prices, "
                    "flight numbers, and departure/arrival times for the cheapest option for each date range "
                    "(if available)."
                ),
                "values": {
                    "origin": {
                        "name": "Suvarnabhumi Airport",
                        "city": "Bangkok",
                        "country": "Thailand",
                        "iata": "BKK",
                        "lat": 13.6810998917,
                        "lon": 100.7470016479,
                        "tz": "Asia/Bangkok",
                        "routeCount": 601,
                        "airlines": {"TG": 0.2596, "PG": 0.0849, "TK": 0.03},
                        "hubScore": 1,
                    },
                    "destination": {
                        "name": "Vienna International Airport",
                        "city": "Vienna",
                        "country": "Austria",
                        "iata": "VIE",
                        "lat": 48.1102981567,
                        "lon": 16.5697002411,
                        "tz": "Europe/Vienna",
                        "routeCount": 396,
                        "airlines": {"OS": 0.5455, "LH": 0.0455, "UA": 0.0303},
                        "hubScore": 1,
                    },
                    "range_one": ["now() + timedelta(days=201)", "now() + timedelta(days=207)"],
                    "range_two": ["now() + timedelta(days=204)", "now() + timedelta(days=210)"],
                    "range_three": ["now() + timedelta(days=207)", "now() + timedelta(days=213)"],
                    "travel_class": {"value": "premium-economy", "label": "Premium economy"},
                    "passengers": 2,
                    "airline": {
                        "code": "NH",
                        "name": "All Nippon Airways",
                        "country": "Japan",
                        "route_count": 547,
                        "importance": 0.2202093398,
                        "hubs": [
                            {"code": "HND", "routes": 60},
                            {"code": "NRT", "routes": 57},
                            {"code": "KIX", "routes": 29},
                            {"code": "CTS", "routes": 25},
                            {"code": "NGO", "routes": 25},
                        ],
                    },
                },
                "location": "San Francisco, CA, United States",
                "timezone": "America/Los_Angeles",
                "timestamp": None,
            }
        ),
        "env": "sim",
        "domain": "noodle_flights",
        "l1_category": "travel",
        "l2_category": "date_range_search_simplified",
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
