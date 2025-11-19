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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Flight":
        cabin_pricing_raw = data.get("cabinPricing", []) or []
        segments_raw = data.get("segments", []) or []
        return cls(
            flightNumber=str(data.get("flightNumber", "")),
            airline=str(data.get("airline", "")),
            airlineName=str(data.get("airlineName", "")),
            aircraft=str(data.get("aircraft", "")),
            depart=data.get("depart"),
            arrive=data.get("arrive"),
            durationMin=(int(data["durationMin"]) if data.get("durationMin") is not None else None),
            distanceKm=(int(data["distanceKm"]) if data.get("distanceKm") is not None else None),
            cabinPricing=[CabinPrice(**cp) for cp in cabin_pricing_raw if isinstance(cp, dict)],
            emissions=(int(data["emissions"]) if data.get("emissions") is not None else None),
            priceUsd=(int(data["priceUsd"]) if data.get("priceUsd") is not None else None),
            originAirport=data.get("originAirport"),
            destinationAirport=data.get("destinationAirport"),
            segments=[FlightSegment(**s) for s in segments_raw if isinstance(s, dict)],
            totalDurationMin=(int(data["totalDurationMin"]) if data.get("totalDurationMin") is not None else None),
            layoverDurationMin=(
                int(data["layoverDurationMin"]) if data.get("layoverDurationMin") is not None else None
            ),
            layoverAirport=data.get("layoverAirport"),
        )


@dataclass
class FlightResults:
    flights: List[Flight]
    expandedOrigins: Optional[List[str]] = None
    expandedDestinations: Optional[List[str]] = None

    @classmethod
    def from_api_response(cls, response: Any) -> "FlightResults":
        if isinstance(response, list):
            flights_list = response
            flights = [Flight.from_dict(f) for f in flights_list if isinstance(f, dict)]
            return cls(flights=flights)
        if isinstance(response, dict):
            flights_list = response.get("departingFlights", []) or []
            flights = [Flight.from_dict(f) for f in flights_list if isinstance(f, dict)]
            return cls(
                flights=flights,
                expandedOrigins=response.get("expandedOrigins"),
                expandedDestinations=response.get("expandedDestinations"),
            )
        raise ValueError(f"Unexpected API response format: {type(response)}. Expected list or dict.")


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
    # Handle trailing Z timezone
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    # Try ISO first
    try:
        dt = datetime.fromisoformat(s)
        return dt.date().isoformat()
    except Exception:
        pass
    # Fallback: try parsing just the date component
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


def verify(
    task: dict, agent_response: Optional[str] = None, site_state: Optional[dict] = None, base_url: Optional[str] = None
):
    """Verify that the agent responded with the formatted total price for the
    cheapest direct roundtrip (departure + return) for the single provided date range.

    Steps:
    - Fetch all departing and returning flights for the origin/destination and date range
    - Filter to direct flights with requested cabin and enough seats
    - Choose the cheapest eligible departure and return
    - Compute total price = (dep_price_per_pax * passengers) + (ret_price_per_pax * passengers)
    - Format as "$X,XXX" and check that it appears in the agent response
    """

    # Parse task values
    values = task.get("values", {})
    origin = values.get("origin", {}).get("iata", "")
    destination = values.get("destination", {}).get("iata", "")
    class_value = values.get("travel_class", {}).get("value", "economy")
    num_passengers = values.get("passengers", 1)
    airline_data = values.get("airline", {})
    airline = Airline(
        code=airline_data.get("code", ""),
        name=airline_data.get("name", ""),
        country=airline_data.get("country", ""),
        route_count=airline_data.get("route_count", 0),
        importance=airline_data.get("importance", 0.0),
        hubs=airline_data.get("hubs", []),
    )

    range_one = values.get("range_one", [])
    if len(range_one) != 2:
        return {"success": False, "verifications": [], "expected": {}, "observed_state": {}}

    dep_str, ret_str = range_one
    # Parse dates from ISO format
    try:
        dep_dt = datetime.fromisoformat(dep_str.replace("Z", "+00:00"))
        ret_dt = datetime.fromisoformat(ret_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return {"success": False, "verifications": [], "expected": {}, "observed_state": {}}

    departure_date = dep_dt.strftime("%Y-%m-%d")
    return_date = ret_dt.strftime("%Y-%m-%d")

    api_response = fetch_flights(
        origin=origin,
        destination=destination,
        departure_date=departure_date,
        return_date=return_date,
        base_url=base_url,
        travel_class=class_value,
        adults=num_passengers,
    )

    departing_flights = [Flight.from_dict(f) for f in api_response.get("departingFlights", [])]
    returning_flights = [Flight.from_dict(f) for f in api_response.get("returningFlights", [])]

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
    ui_dep_raw = (site_state or {}).get("departureDate")
    ui_ret_raw = (site_state or {}).get("returnDate")

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
        "observed_state": (site_state or {}),
    }


# =========================
# Rendering helpers
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


def _format_date_display(date_str: str) -> str:
    """Convert YYYY-MM-DD to 'Month DD, YYYY' for display."""
    dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    return dt.strftime("%B %d, %Y")


def _render_values(values: dict[str, Any], timezone: str, timestamp: int) -> dict[str, Any]:
    """Render range_one date formulas for natural_airline_search_simple tasks.

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
    """Replace {departureDate}/{returnDate} placeholders with formatted dates.

    Example:
        Input task: "Find flights departing on {departureDate} and returning on {returnDate}..."
        Input values: {"range_one": ["2025-12-04T00:00:00+00:00", "2025-12-07T00:00:00+00:00"]}
        Output: "Find flights departing on December 04, 2025 and returning on December 07, 2025..."
    """
    result = task

    # Replace {departureDate} and {returnDate} placeholders
    range_one = values.get("range_one")
    if isinstance(range_one, list) and len(range_one) == 2:
        dep_str, ret_str = range_one
        # Parse ISO format strings to datetime objects for formatting
        try:
            if "{departureDate}" in result:
                dep_display = _format_date_display(dep_str)
                result = result.replace("{departureDate}", dep_display)
            if "{returnDate}" in result:
                ret_display = _format_date_display(ret_str)
                result = result.replace("{returnDate}", ret_display)
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
    eval_target = get_import_path(HalluminateNoodleFlightsNaturalAirlineSearchSimple)
    eval_config = {"_target_": eval_target, "task": task, "values": values, "user_metadata": user_metadata.model_dump()}

    # Create the task config
    return BaseTaskConfig(
        task=task,
        url=url,
        user_metadata=user_metadata,
        eval_config=eval_config,
    )


# =========================
# Evaluator
# =========================


class InputDict(TypedDict, total=False):
    page: "Page"
    answer_message: str


class FinalResult(BaseModel):
    """Result model for Noodle flights natural airline search simple task."""

    score: float
    reasoning: str
    current_state: dict[str, Any]
    ground_truth: dict[str, Any]


class HalluminateNoodleFlightsNaturalAirlineSearchSimple(BaseMetric):
    """Evaluator that runs verifier for Noodle flights natural airline search simple task."""

    def __init__(self, task: str, values: dict[str, Any], user_metadata: dict[str, Any]) -> None:
        super().__init__()
        self.task = task
        self.values = values
        self.user_metadata = user_metadata

        self._task_dict = {"id": "", "task": self.task, "values": self.values}
        self._ground_truth = values
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

        result = verify(self._task_dict, self._answer_message, current_state, None)

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


if __name__ == "__main__":
    from westworld.base import DatasetItem, instantiate

    dataset_row = {
        "task_id": "halluminate_bench/noodle_flights/natural_airline_search_simple/101",
        "task_generation_config_json": json.dumps(
            {
                "_target_": "halluminate_bench.noodle_flights.natural_airline_search_simple.generate_task_config",
                "url": "https://flights.halluminate.ai/?apiKey={HALLUMINATE_API_KEY}",
                "task": (
                    "I need to find out the complete fare for 4 adult passengers flying economy class on a nonstop"
                    " round-trip journey from Sydney (SYD) to London Gatwick (LGW), departing on {departureDate} and"
                    " returning on {returnDate} - what would be the total amount?"
                ),
                "values": {
                    "origin": {
                        "name": "Sydney Kingsford Smith International Airport",
                        "city": "Sydney",
                        "country": "Australia",
                        "iata": "SYD",
                        "lat": -33.9460983276,
                        "lon": 151.1770019531,
                        "tz": "Australia/Sydney",
                        "routeCount": 387,
                        "airlines": {
                            "QF": 0.2351,
                            "VA": 0.1602,
                            "JQ": 0.1137,
                            "ZL": 0.0879,
                            "AA": 0.062,
                            "NZ": 0.0465,
                            "EK": 0.0362,
                            "MU": 0.031,
                            "UA": 0.0155,
                            "AI": 0.0103,
                            "TK": 0.0103,
                            "CI": 0.0103,
                            "CA": 0.0103,
                            "LA": 0.0103,
                            "FJ": 0.0103,
                            "GA": 0.0103,
                            "TZ": 0.0052,
                            "OZ": 0.0052,
                            "VN": 0.0052,
                            "PR": 0.0052,
                            "SA": 0.0052,
                            "PX": 0.0052,
                            "TN": 0.0052,
                            "SB": 0.0052,
                            "TG": 0.0052,
                            "SQ": 0.0052,
                            "3U": 0.0052,
                            "NF": 0.0052,
                            "MH": 0.0052,
                            "AC": 0.0052,
                            "AY": 0.0052,
                            "BA": 0.0052,
                            "CX": 0.0052,
                            "CZ": 0.0052,
                            "D7": 0.0052,
                            "DL": 0.0052,
                            "ET": 0.0052,
                            "EY": 0.0052,
                            "HA": 0.0052,
                            "IB": 0.0052,
                            "JL": 0.0052,
                            "KE": 0.0052,
                            "VS": 0.0026,
                            "KL": 0.0026,
                        },
                        "hubScore": 1,
                    },
                    "destination": {
                        "name": "London Gatwick Airport",
                        "city": "London",
                        "country": "United Kingdom",
                        "iata": "LGW",
                        "lat": 51.148102,
                        "lon": -0.190278,
                        "tz": "Europe/London",
                        "routeCount": 622,
                        "airlines": {
                            "U2": 0.3248,
                            "BA": 0.1383,
                            "TO": 0.1318,
                            "DY": 0.0965,
                            "TOM": 0.0595,
                            "IB": 0.0418,
                            "TCX": 0.037,
                            "VS": 0.0289,
                            "FR": 0.0161,
                            "TS": 0.0129,
                            "AY": 0.0096,
                            "TP": 0.0096,
                            "EI": 0.0096,
                            "AA": 0.0096,
                            "TK": 0.0064,
                            "S4": 0.0064,
                            "3O": 0.0064,
                            "DL": 0.0064,
                            "DE": 0.0064,
                            "AT": 0.0064,
                            "GR": 0.0032,
                            "OU": 0.0032,
                            "QF": 0.0032,
                            "FI": 0.0032,
                            "EK": 0.0032,
                            "CA": 0.0032,
                            "BW": 0.0032,
                            "BT": 0.0032,
                            "B2": 0.0032,
                            "UX": 0.0032,
                            "XY": 0.0032,
                        },
                        "hubScore": 1,
                    },
                    "range_one": ["now() + timedelta(days=316)", "now() + timedelta(days=316) + timedelta(days=3)"],
                    "travel_class": {"value": "economy", "label": "Economy"},
                    "passengers": 4,
                    "airline": {
                        "code": "AA",
                        "name": "American Airlines",
                        "country": "United States",
                        "route_count": 2354,
                        "importance": 0.9476650564,
                        "hubs": [
                            {"code": "DFW", "routes": 183},
                            {"code": "CLT", "routes": 134},
                            {"code": "ORD", "routes": 124},
                            {"code": "MIA", "routes": 120},
                            {"code": "PHL", "routes": 118},
                            {"code": "LHR", "routes": 94},
                            {"code": "DCA", "routes": 75},
                            {"code": "PHX", "routes": 74},
                            {"code": "LAX", "routes": 69},
                            {"code": "JFK", "routes": 61},
                        ],
                    },
                },
                "location": "Sydney, Australia",
                "timezone": "Australia/Sydney",
            }
        ),
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
