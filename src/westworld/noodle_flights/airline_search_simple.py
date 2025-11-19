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


def _format_price_usd(amount: Optional[int]) -> Optional[str]:
    if amount is None:
        return None
    return f"${amount:,}"


def _normalize_date_yyyy_mm_dd(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    s = str(value).strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
        return dt.date().isoformat()
    except Exception:
        pass
    try:
        dt = datetime.strptime(s.split("T")[0], "%Y-%m-%d")
        return dt.date().isoformat()
    except Exception:
        return None


def _per_passenger_price_usd(flight: Optional[Flight], cabin_name: str) -> Optional[int]:
    if not flight:
        return None
    cabin_price = _find_cabin_price_usd(flight, cabin_name)
    if isinstance(cabin_price, int):
        return cabin_price
    return _effective_price_usd(flight, cabin_name)


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
    return {
        "flight_number": flight.flightNumber,
        "airline_name": flight.airlineName,
        "departure_time": depart_time,
        "arrival_time": arrive_time,
        "source_airport_code": origin_code,
        "destination_airport_code": destination_code,
        "total_price": total_price,
    }


def filter_flights(
    flights: List[Flight],
    class_name: str,
    num_passengers: int,
    airline: Airline,
) -> List[Flight]:
    """
    Filter flights for direct service, cabin availability and seats, then return:
    - the cheapest direct option, and
    - all direct options operated by the specified airline
    (without duplicates), ordered by price ascending.
    """

    cabin = _normalize_cabin_name(class_name)

    def has_cabin_and_seats(f: Flight) -> bool:
        for cp in f.cabinPricing:
            if cp.cabin == cabin and cp.seatsRemaining >= num_passengers:
                return True
        return False

    direct_flights = [f for f in flights if not f.segments]
    eligible = [f for f in direct_flights if has_cabin_and_seats(f)]

    # Sort by effective price (missing price -> very high)
    def price_key(f: Flight) -> int:
        p = _effective_price_usd(f, cabin)
        return p if p is not None else 10**9

    eligible.sort(key=price_key)

    result: List[Flight] = []
    seen: set[str] = set()

    if eligible:
        cheapest = eligible[0]
        if cheapest.flightNumber not in seen:
            result.append(cheapest)
            seen.add(cheapest.flightNumber)

    for f in eligible:
        if f.airline == airline.code and f.flightNumber not in seen:
            result.append(f)
            seen.add(f.flightNumber)

    return result


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
    travel_class = TravelClass(**data["travel_class"])
    passengers = int(data["passengers"])
    airline = parse_airline(data["airline"])

    return TaskData(
        origin=origin,
        destination=destination,
        range_one=range_one,
        travel_class=travel_class,
        passengers=passengers,
        airline=airline,
    )


def verify_airline_search_complete(
    current_state: Dict[str, Any], task_data: TaskData, agent_response: Optional[str] = None
) -> Dict[str, Any]:
    """Verify that the agent responded with the formatted total price for the
    cheapest direct roundtrip (departure + return) for the single provided date range.

    Steps:
    - Fetch all departing and returning flights for the origin/destination and date range
    - Filter to direct flights with requested cabin and enough seats
    - Choose the cheapest eligible departure and return
    - Compute total price = (dep_price_per_pax * passengers) + (ret_price_per_pax * passengers)
    - Format as "$X,XXX" and check that it appears in the agent response
    """

    origin = task_data.origin.iata
    destination = task_data.destination.iata
    class_value = task_data.travel_class.value
    num_passengers = task_data.passengers
    airline = task_data.airline

    dep_dt, ret_dt = task_data.range_one
    departure_date = dep_dt.strftime("%Y-%m-%d")
    return_date = ret_dt.strftime("%Y-%m-%d")

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

    # Reuse filter to get cheapest eligible first
    filtered_departing = filter_flights(departing_flights, class_value, num_passengers, airline)
    filtered_returning = filter_flights(returning_flights, class_value, num_passengers, airline)

    cheapest_dep = filtered_departing[0] if filtered_departing else None
    cheapest_ret = filtered_returning[0] if filtered_returning else None

    cabin = _normalize_cabin_name(class_value)
    dep_price_per_pax = _per_passenger_price_usd(cheapest_dep, cabin)
    ret_price_per_pax = _per_passenger_price_usd(cheapest_ret, cabin)

    dep_total = dep_price_per_pax * num_passengers if isinstance(dep_price_per_pax, int) else None
    ret_total = ret_price_per_pax * num_passengers if isinstance(ret_price_per_pax, int) else None
    total_price = (dep_total + ret_total) if isinstance(dep_total, int) and isinstance(ret_total, int) else None
    formatted_total_price = _format_price_usd(total_price)

    # Detect an incorrect agent pattern where the already-totaled price is multiplied by passengers again
    doubled_total_price = (
        total_price * num_passengers
        if isinstance(total_price, int) and isinstance(num_passengers, int) and num_passengers > 1
        else None
    )
    formatted_doubled_total_price = _format_price_usd(doubled_total_price)

    expected: Dict[str, Any] = {
        "cheapest_departure_flight": (
            _build_expected_flight(cheapest_dep, num_passengers, cabin) if cheapest_dep else None
        ),
        "cheapest_return_flight": (
            _build_expected_flight(cheapest_ret, num_passengers, cabin) if cheapest_ret else None
        ),
        "total_price": total_price,
        "formatted_total_price": formatted_total_price,
    }

    response_text = agent_response or ""
    price_present = bool(formatted_total_price) and (formatted_total_price in response_text)
    doubled_present = bool(formatted_doubled_total_price) and (formatted_doubled_total_price in response_text)

    # Validate UI state dates if provided
    ui_dep_raw = current_state.get("departureDate")
    ui_ret_raw = current_state.get("returnDate")

    ui_dep = _normalize_date_yyyy_mm_dd(ui_dep_raw)
    ui_ret = _normalize_date_yyyy_mm_dd(ui_ret_raw)

    dep_date_ok = ui_dep == departure_date
    ret_date_ok = ui_ret == return_date

    verifications: List[Dict[str, Any]] = [
        {"bucket": "total_price", "candidate": formatted_total_price, "matched": bool(price_present)},
        {
            "bucket": "doubled_total_absent",
            "candidate": formatted_doubled_total_price,
            "matched": bool(not doubled_present),
        },
        {"bucket": "ui_departure_date", "candidate": ui_dep, "matched": bool(dep_date_ok)},
        {"bucket": "ui_return_date", "candidate": ui_ret, "matched": bool(ret_date_ok)},
    ]

    success = bool(price_present and (not doubled_present) and dep_date_ok and ret_date_ok)

    return {
        "success": success,
        "verifications": verifications,
        "expected": expected,
        "observed_state": current_state,
    }


# =========================
# Evaluator
# =========================


class InputDict(TypedDict, total=False):
    page: "Page"
    answer_message: str


class FinalResult(BaseModel):
    """Result model for Noodle flights airline search simple task."""

    score: float
    reasoning: str
    current_state: dict[str, Any]
    ground_truth: TaskData


class HalluminateNoodleFlightsAirlineSearchSimple(BaseMetric):
    """Evaluator that runs verifier for Noodle flights airline search simple task."""

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

        result = verify_airline_search_complete(current_state, self._ground_truth, self._answer_message)

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
                + str(result["expected"])
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
        Input:  formula_str="now() + timedelta(days=1)", timezone="America/Los_Angeles", timestamp=1715702400
        Output: datetime(2024, 5, 15, 9, 0, tzinfo=ZoneInfo('America/Los_Angeles'))

        Input:  formula_str="now() + timedelta(days=1)", timezone="Asia/Tokyo", timestamp=1715702400
        Output: datetime(2024, 5, 16, 1, 0, tzinfo=ZoneInfo('Asia/Tokyo'))
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
    """Render range_one date formulas for airline_search_simple tasks.

    Only render fields that contain date formulas (range_one).
    All other fields are left unchanged.
    """
    evaluated = dict(values)

    # Evaluate range_one formulas if present
    range_one = evaluated.get("range_one")
    if isinstance(range_one, list) and len(range_one) == 2:
        dep_formula = range_one[0]
        ret_formula = range_one[1]

        # Check if they are formulas (strings containing "now()" or "timedelta")
        if isinstance(dep_formula, str) and ("now()" in dep_formula or "timedelta" in dep_formula):
            dep_dt = _evaluate_date_formula(dep_formula, timezone, timestamp)
            ret_dt = _evaluate_date_formula(ret_formula, timezone, timestamp)
            # Store as ISO format strings
            evaluated["range_one"] = [dep_dt.isoformat(), ret_dt.isoformat()]

    return evaluated


def _render_task(task: str, values: dict[str, Any]) -> str:
    """Replace {dateRange} placeholder with formatted date range.

    Example:
        Input task: "Find the cheapest flight on {dateRange}..."
        Input values: {"range_one": ["2025-12-04T00:00:00", "2025-12-07T00:00:00"]}
        Output: "Find the cheapest flight on December 4-7, 2025..."
    """
    result = task

    # Replace {dateRange} placeholder
    if "{dateRange}" in result:
        range_one = values.get("range_one")
        if isinstance(range_one, list) and len(range_one) == 2:
            dep_str, ret_str = range_one
            # Parse ISO format strings to datetime objects for formatting
            try:
                dep_dt = datetime.fromisoformat(dep_str.replace("Z", "+00:00"))
                ret_dt = datetime.fromisoformat(ret_str.replace("Z", "+00:00"))
                date_range_str = _format_date_range(dep_dt, ret_dt)
                result = result.replace("{dateRange}", date_range_str)
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
    eval_target = get_import_path(HalluminateNoodleFlightsAirlineSearchSimple)
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
        "task_id": "halluminate_bench/noodle_flights/airline_search_simple/1",
        "task_generation_config_json": json.dumps(
            {
                "_target_": "halluminate_bench.noodle_flights.airline_search_simple.generate_task_config",
                "url": "https://flights.halluminate.ai/?apiKey={HALLUMINATE_API_KEY}",
                "task": (
                    "Search for direct Business class flights from CPH to BCN for 4 adult passengers for the date range"
                    " {dateRange}. What is the total price for the cheapest departure and return flight?"
                ),
                "values": {
                    "origin": {
                        "name": "Copenhagen Kastrup Airport",
                        "city": "Copenhagen",
                        "country": "Denmark",
                        "iata": "CPH",
                        "lat": 55.6179008484,
                        "lon": 12.6560001373,
                        "tz": "Europe/Copenhagen",
                        "routeCount": 413,
                        "airlines": {
                            "SK": 0.3923,
                            "DY": 0.1985,
                            "U2": 0.0581,
                            "AC": 0.0266,
                            "LH": 0.0266,
                            "IB": 0.0242,
                            "WF": 0.0194,
                            "AY": 0.0194,
                            "CA": 0.0145,
                            "TG": 0.0145,
                            "LX": 0.0097,
                            "PK": 0.0097,
                            "HU": 0.0097,
                            "ET": 0.0097,
                            "PC": 0.0097,
                            "SN": 0.0048,
                            "OU": 0.0048,
                            "SQ": 0.0048,
                            "S4": 0.0048,
                            "SU": 0.0048,
                            "RJ": 0.0048,
                            "RC": 0.0048,
                            "QS": 0.0048,
                            "TK": 0.0048,
                            "QR": 0.0048,
                            "QF": 0.0048,
                            "TP": 0.0048,
                            "A3": 0.0048,
                            "OS": 0.0048,
                            "MU": 0.0048,
                            "AF": 0.0048,
                            "AT": 0.0048,
                            "BA": 0.0048,
                            "BT": 0.0048,
                            "DX": 0.0048,
                            "EI": 0.0048,
                            "EK": 0.0048,
                            "FI": 0.0048,
                            "GL": 0.0048,
                            "HV": 0.0048,
                            "JU": 0.0048,
                            "KL": 0.0048,
                            "LG": 0.0048,
                            "LO": 0.0048,
                            "AA": 0.0048,
                            "MS": 0.0048,
                            "XQ": 0.0048,
                            "UA": 0.0024,
                        },
                        "hubScore": 1,
                    },
                    "destination": {
                        "name": "Barcelona International Airport",
                        "city": "Barcelona",
                        "country": "Spain",
                        "iata": "BCN",
                        "lat": 41.2971,
                        "lon": 2.07846,
                        "tz": "Europe/Madrid",
                        "routeCount": 537,
                        "airlines": {
                            "IB": 0.3836,
                            "FR": 0.1453,
                            "U2": 0.0521,
                            "DY": 0.041,
                            "W6": 0.0372,
                            "UX": 0.0298,
                            "BA": 0.0149,
                            "LH": 0.0149,
                            "AA": 0.0149,
                            "3O": 0.0112,
                            "AY": 0.0112,
                            "LX": 0.0112,
                            "LS": 0.0112,
                            "SK": 0.0112,
                            "HV": 0.0112,
                            "SQ": 0.0112,
                            "AH": 0.0074,
                            "QR": 0.0074,
                            "S4": 0.0074,
                            "KL": 0.0074,
                            "AF": 0.0074,
                            "PK": 0.0074,
                            "SU": 0.0074,
                            "EI": 0.0074,
                            "TP": 0.0074,
                            "TS": 0.0074,
                            "AT": 0.0074,
                            "PC": 0.0037,
                            "TK": 0.0037,
                            "RO": 0.0037,
                            "RJ": 0.0037,
                            "TU": 0.0037,
                            "AC": 0.0037,
                            "UA": 0.0037,
                            "QS": 0.0037,
                            "SN": 0.0037,
                            "AV": 0.0037,
                            "OU": 0.0037,
                            "OS": 0.0037,
                            "B2": 0.0037,
                            "BT": 0.0037,
                            "CA": 0.0037,
                            "CX": 0.0037,
                            "DL": 0.0037,
                            "AR": 0.0037,
                            "EK": 0.0037,
                            "ET": 0.0037,
                            "HU": 0.0037,
                            "IZ": 0.0037,
                            "LG": 0.0037,
                            "A3": 0.0037,
                            "LO": 0.0037,
                            "MS": 0.0037,
                            "MU": 0.0037,
                            "LY": 0.0037,
                            "EB": 0.0019,
                        },
                        "hubScore": 1,
                    },
                    "range_one": ["now() + timedelta(days=291)", "now() + timedelta(days=295)"],
                    "travel_class": {"value": "business", "label": "Business"},
                    "passengers": 4,
                    "airline": {
                        "code": "LH",
                        "name": "Lufthansa Cargo",
                        "country": "Germany",
                        "route_count": 923,
                        "importance": 0.3715780998,
                        "hubs": [
                            {"code": "FRA", "routes": 171},
                            {"code": "MUC", "routes": 129},
                            {"code": "DUS", "routes": 38},
                            {"code": "BRU", "routes": 22},
                            {"code": "EWR", "routes": 22},
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
        "l2_category": "airline_search_simple",
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
