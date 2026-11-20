"""The answer checks through the metric classes: update() with a page and an answer, then compute()."""

import asyncio

from tests.noodle_flights.flight_data import flight
from tests.noodle_flights.test_airline_and_date_range_answers import API, STATE, airline_values, date_range_values
from westworld.base import UserMetadata
from westworld.noodle_flights import (
    airline_search_simple,
    date_range_search,
    date_range_search_simplified,
    flight_search_cheapest,
    flight_search_roundtrip,
    natural_airline_search_simple,
)


class FakePage:
    """Stands in for the Playwright page; returns the simulator's testing-API state."""

    def __init__(self, state: dict):
        self.state = state

    async def evaluate(self, script: str) -> dict:
        return self.state


def score(evaluator, state: dict, answer: str) -> float:
    async def run():
        await evaluator.update(page=FakePage(state), answer_message=answer)
        return await evaluator.compute()

    return asyncio.run(run()).score


def test_cheapest_evaluator(monkeypatch):
    flights = [flight("AA108/AA985", 247), flight("UA835/WS935", 268), flight("UA389", 284)]
    monkeypatch.setattr(
        flight_search_cheapest, "query_noodle_flights_simulator", lambda *args, **kwargs: {"departingFlights": flights}
    )
    evaluator = flight_search_cheapest.HalluminateNoodleFlightsFlightSearchCheapest(
        task="Find the cheapest one-way flight from SFO to JFK on June 12, 2030 and return its flight number and price.",
        values={"origin": "SFO", "destination": "JFK", "date": "2030-06-12", "tripType": "one-way", "class": "economy"},
        user_metadata=UserMetadata(),
    )
    state = {
        "tripType": "one-way",
        "fromLocation": "SFO",
        "toLocation": "JFK",
        "departureDate": "2030-06-12",
        "travelClass": "economy",
        "isSearchReady": True,
    }
    assert score(evaluator, state, "The cheapest flight is AA108/AA985 at $247.00.") == 1.0
    assert score(evaluator, state, "AA108/AA985 $247, UA835/WS935 $268, UA389 $284") == 0.0
    assert score(evaluator, {**state, "toLocation": "EWR"}, "The cheapest flight is AA108/AA985 at $247.") == 0.0


def test_roundtrip_evaluator(monkeypatch):
    data = {
        "departingFlights": [flight("B6123", 225), flight("DL400", 260)],
        "returningFlights": [flight("B6124", 240), flight("DL401", 250)],
    }
    monkeypatch.setattr(flight_search_roundtrip, "query_noodle_flights_simulator_roundtrip", lambda *args, **kwargs: data)
    evaluator = flight_search_roundtrip.HalluminateNoodleFlightsFlightSearchRoundtrip(
        task="Find the cheapest round-trip flight option from LAX to BOS. Return the total price and the flight numbers.",
        values={
            "origin": "LAX",
            "destination": "BOS",
            "departureDate": "2030-09-10",
            "returnDate": "2030-09-17",
            "tripType": "round-trip",
            "class": "economy",
        },
        user_metadata=UserMetadata(),
    )
    state = {
        "tripType": "round-trip",
        "fromLocation": "LAX",
        "toLocation": "BOS",
        "departureDate": "2030-09-10",
        "returnDate": "2030-09-17",
        "travelClass": "economy",
        "isSearchReady": True,
    }
    assert score(evaluator, state, "Outbound B6123, return B6124, total $465.") == 1.0
    assert score(evaluator, state, "B6123 $225, DL400 $260; B6124 $240, DL401 $250. Total $465.") == 0.0


def test_airline_evaluators(monkeypatch):
    for module in (airline_search_simple, natural_airline_search_simple):
        monkeypatch.setattr(module, "fetch_flights", lambda *args, **kwargs: API)
    simple = airline_search_simple.HalluminateNoodleFlightsAirlineSearchSimple(
        task="What is the total price for the cheapest departure and return flight?",
        values=airline_values(),
        user_metadata=UserMetadata(),
    )
    natural = natural_airline_search_simple.HalluminateNoodleFlightsNaturalAirlineSearchSimple(
        task="What would be the total amount?", values=airline_values(), user_metadata=UserMetadata().model_dump()
    )
    for evaluator in (simple, natural):
        assert score(evaluator, STATE, "The total for 2 adults is $2,200.") == 1.0
        assert score(evaluator, STATE, "OS101 500, UA200 520, OS102 600, UA201 640. Total $2,200") == 0.0


def test_date_range_evaluators(monkeypatch):
    for module in (date_range_search, date_range_search_simplified):
        monkeypatch.setattr(module, "fetch_flights", lambda *args, **kwargs: API)
    full = date_range_search.HalluminateNoodleFlightsDateRangeSearch(
        task="Extract the cheapest option and all United options for each date range.",
        values=date_range_values(),
        user_metadata=UserMetadata(),
    )
    simplified = date_range_search_simplified.HalluminateNoodleFlightsDateRangeSearchSimplified(
        task="Extract the cheapest option for each date range.", values=date_range_values(), user_metadata=UserMetadata()
    )
    assert score(full, STATE, "Each range: cheapest OS101 and OS102; United UA200 and UA201.") == 1.0
    assert score(full, STATE, "Each range: OS101, OS102, UA200, UA201 and AF700.") == 0.0
    assert score(simplified, STATE, "Each range: OS101 for $500 out and OS102 for $600 back.") == 1.0
    assert score(simplified, STATE, "Each range: OS101 and OS102, or UA200 and UA201.") == 0.0
