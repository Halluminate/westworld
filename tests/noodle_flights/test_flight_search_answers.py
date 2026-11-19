"""Answer checks for the cheapest one-way and cheapest round-trip Noodle Flights tasks."""

import pytest

from tests.noodle_flights.flight_data import flight
from westworld.noodle_flights import flight_search_cheapest, flight_search_roundtrip


ONE_WAY = [
    flight("AA108/AA985", 247),
    flight("UA835/WS935", 268),
    flight("UA389", 284),
    flight("DL565", 284),
    flight("B6101", 301),
]


@pytest.fixture
def one_way(monkeypatch):
    monkeypatch.setattr(
        flight_search_cheapest, "query_noodle_flights_simulator", lambda *args, **kwargs: {"departingFlights": ONE_WAY}
    )


def check_one_way(answer: str, direct: bool = False):
    task = "Find the cheapest one-way {}flight from SFO to JFK on June 12, 2030 and return its flight number and price."
    task_data = {"values": {"origin": "SFO", "destination": "JFK", "date": "2030-06-12"}, "task": task.format("direct " if direct else "")}
    return flight_search_cheapest.verify_agent_response(answer, "task", task_data)


@pytest.mark.parametrize(
    "answer",
    [
        "The cheapest flight is AA108/AA985 at $247.",
        "The cheapest flight is AA108/AA985 at $247.00.",
        "AA108/AA985, USD 247",
        "Take AA 108 connecting to AA 985 for 247 dollars.",
    ],
)
def test_cheapest_answers_in_common_formats_pass(one_way, answer):
    assert check_one_way(answer).passed


@pytest.mark.parametrize(
    "answer",
    [
        "Here are all the flights: " + "; ".join(f"{f['flightNumber']} ${f['priceUsd']}" for f in ONE_WAY),
        "I would skip AA108/AA985 ($247) because of the layover; book UA835/WS935 at $268.",
        "The cheapest flight is UA835/WS935 at $247.",
        "The cheapest flight is AA108/AA985.",
    ],
)
def test_lists_and_wrong_answers_fail(one_way, answer):
    result = check_one_way(answer)
    assert not result.passed


def test_naming_other_flights_is_reported(one_way):
    result = check_one_way("AA108/AA985 at $247, or UA389 at $284.")
    assert not result.passed
    assert result.details["otherFlightsMentioned"] == ["UA389"]


def test_any_flight_tied_at_the_cheapest_direct_price_passes(one_way):
    assert check_one_way("UA389 at $284.", direct=True).passed
    assert check_one_way("DL565 at $284.", direct=True).passed
    assert check_one_way("UA389 or DL565, both $284.", direct=True).passed
    assert not check_one_way("AA108/AA985 at $247.", direct=True).passed


ROUND_TRIP = {
    "departingFlights": [flight("AA258/AC224", 225), flight("B6123", 225), flight("DL400", 260)],
    "returningFlights": [flight("AA198/AC116", 216), flight("B6124", 240)],
}


@pytest.fixture
def round_trip(monkeypatch):
    monkeypatch.setattr(
        flight_search_roundtrip, "query_noodle_flights_simulator_roundtrip", lambda *args, **kwargs: ROUND_TRIP
    )


def check_round_trip(answer: str):
    task_data = {
        "values": {"origin": "LAX", "destination": "BOS", "departureDate": "2030-09-10", "returnDate": "2030-09-17"},
        "task": "Find the cheapest round-trip flight option from LAX to BOS.",
    }
    return flight_search_roundtrip.verify_agent_response(answer, "task", task_data)


@pytest.mark.parametrize(
    "answer",
    [
        "Outbound AA258/AC224, return AA198/AC116, total $441.",
        "Outbound B6123 and return AA198/AC116 for $441.00 in total.",
    ],
)
def test_cheapest_round_trip_answers_pass(round_trip, answer):
    assert check_round_trip(answer).passed


@pytest.mark.parametrize(
    "answer",
    [
        "Options: AA258/AC224 $225, B6123 $225, DL400 $260; returns AA198/AC116 $216, B6124 $240. Total $441.",
        "Outbound AA258/AC224, return B6124, total $441.",
        "Outbound AA258/AC224, return AA198/AC116, total $465.",
    ],
)
def test_round_trip_lists_and_wrong_answers_fail(round_trip, answer):
    assert not check_round_trip(answer).passed
