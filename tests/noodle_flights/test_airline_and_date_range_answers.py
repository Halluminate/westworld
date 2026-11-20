"""Answer checks for the airline-search and date-range Noodle Flights tasks."""

import pytest

from tests.noodle_flights.flight_data import airline, airport, flight
from westworld.noodle_flights import (
    airline_search_simple,
    date_range_search,
    date_range_search_simplified,
    natural_airline_search_simple,
)


OUTBOUND = [flight("OS101", 500), flight("UA200", 520), flight("AF700", 700), flight("LH300/LH301", 450)]
RETURN = [flight("OS102", 600), flight("UA201", 640)]
API = {"departingFlights": OUTBOUND, "returningFlights": RETURN}
STATE = {"departureDate": "2030-06-12", "returnDate": "2030-06-19"}
RANGE = ["2030-06-12T00:00:00", "2030-06-19T00:00:00"]

# Two passengers: (500 + 600) x 2 = $2,200; the already-doubled mistake would be $4,400.
CORRECT_TOTALS = [
    "The total for 2 adults is $2,200.",
    "$2200.00 in total",
    "Total: 2,200 USD",
    "Outbound $500 and return $600 per person, $2,200 in total.",
    "OS101 and OS102 are the cheapest direct flights; together they cost $2,200 for both of you.",
    "Total $2,200. A one-stop LH300/LH301 would be $900 for both, but it is not direct.",
]
WRONG_TOTALS = [
    "OS101 $500, UA200 $520, OS102 $600, UA201 $640. The total is $2,200.",
    "OS101 500, UA200 520, OS102 600, UA201 640. Total $2,200",
    "Options: 500, 520, 600 and 640 per person; the total is $2,200.",
    "That is $2,200 per person, so $4,400 for both of you.",
    "The total for 2 adults is $2,000.",
    "The total is 2,200.",
    "Flight OS2200 costs $9,999.",
    "Total $2,200, with UA200 as a backup.",
    # Quoting another direct option, even as an aside, counts as listing alternatives.
    "Total $2,200. The next cheapest outbound was $520.",
]


@pytest.fixture
def simulator(monkeypatch):
    for module in (airline_search_simple, natural_airline_search_simple, date_range_search, date_range_search_simplified):
        monkeypatch.setattr(module, "fetch_flights", lambda *args, **kwargs: API)


def airline_values(**extra):
    return {
        "origin": airport("VIE"),
        "destination": airport("MCO"),
        "range_one": RANGE,
        "travel_class": {"value": "economy", "label": "Economy"},
        "passengers": 2,
        "airline": airline("OS"),
        **extra,
    }


def check_airline(answer: str, state=STATE):
    task_data = airline_search_simple.create_task_data_from_dict(airline_values())
    return airline_search_simple.verify_airline_search_complete(state, task_data, answer)["success"]


def check_natural(answer: str):
    return natural_airline_search_simple.verify({"values": airline_values()}, answer, STATE, None)["success"]


@pytest.mark.parametrize("answer", CORRECT_TOTALS)
def test_airline_total_in_common_formats_passes(simulator, answer):
    assert check_airline(answer)
    assert check_natural(answer)


@pytest.mark.parametrize("answer", WRONG_TOTALS)
def test_airline_lists_and_wrong_totals_fail(simulator, answer):
    assert not check_airline(answer)
    assert not check_natural(answer)


def test_airline_search_still_checks_the_dates(simulator):
    assert not check_airline("$2,200", state={"departureDate": "2030-06-13", "returnDate": "2030-06-19"})


def date_range_values(airline_code: str = "UA"):
    return {
        "origin": airport("SEA"),
        "destination": airport("SYD"),
        "range_one": RANGE,
        "range_two": ["2030-06-15T00:00:00", "2030-06-22T00:00:00"],
        "range_three": ["2030-06-18T00:00:00", "2030-06-25T00:00:00"],
        "travel_class": {"value": "premium-economy", "label": "Premium economy"},
        "passengers": 1,
        "airline": airline(airline_code),
    }


def check_date_range(answer: str):
    task_data = date_range_search.create_task_data_from_dict(date_range_values())
    return date_range_search.verify_date_range_search_complete(STATE, task_data, answer)["success"]


def check_date_range_simplified(answer: str):
    task_data = date_range_search_simplified.create_task_data_from_dict(date_range_values())
    return date_range_search_simplified.verify_date_range_search_simplified_complete(STATE, task_data, answer)[
        "success"
    ]


@pytest.mark.parametrize(
    "answer",
    [
        # Cheapest direct flights are OS101/OS102; the United options are UA200/UA201.
        "Each range: cheapest OS101 ($500) and OS102 ($600); United options UA200 ($520) and UA201 ($640).",
        "each range: cheapest os101 ($500) and os102 ($600); united ua200 ($520) and ua201 ($640).",
        "Each range: cheapest OS101 and OS102; United UA200 and UA201; a one-stop LH300/LH301 is $450.",
    ],
)
def test_date_range_answers_with_the_requested_flights_pass(simulator, answer):
    assert check_date_range(answer)


def test_date_range_answer_listing_other_flights_fails(simulator):
    answer = "Each range: cheapest OS101 ($500) and OS102 ($600); United UA200 and UA201; also AF700 ($700)."
    assert not check_date_range(answer)
    assert not check_date_range("Each range: OS101 and OS102 are the cheapest.")


def test_date_range_accepts_any_flight_tied_at_the_cheapest_price(monkeypatch):
    api = {"departingFlights": [*OUTBOUND, flight("DL77", 500)], "returningFlights": RETURN}
    for module in (date_range_search, date_range_search_simplified):
        monkeypatch.setattr(module, "fetch_flights", lambda *args, **kwargs: api)
    assert check_date_range("Each range: cheapest DL77 ($500) and OS102 ($600); United UA200 and UA201.")
    assert check_date_range_simplified("Each range: cheapest DL77 and OS102.")


def test_date_range_answer_may_name_any_flight_of_the_requested_airline(monkeypatch):
    # UA210 has too few seats for two passengers and UA300/UA301 has a stop, so the verifier
    # does not require them, but naming them as United options is not a wrong answer.
    api = {
        "departingFlights": [*OUTBOUND, flight("UA210", 530, seats=1), flight("UA300/UA301", 470)],
        "returningFlights": RETURN,
    }
    monkeypatch.setattr(date_range_search, "fetch_flights", lambda *args, **kwargs: api)
    task_data = date_range_search.create_task_data_from_dict({**date_range_values(), "passengers": 2})
    answer = (
        "Each range: cheapest OS101 ($1,000) and OS102 ($1,200); United UA200 and UA201, "
        "plus UA210 (one seat left) and UA300/UA301 (one stop)."
    )
    assert date_range_search.verify_date_range_search_complete(STATE, task_data, answer)["success"]


@pytest.mark.parametrize(
    "answer",
    [
        "Each range: OS101 for $500 out and OS102 for $600 back.",
        "Each range: outbound $500.00 and return $600.",
        "each range: os101 out and os102 back.",
    ],
)
def test_simplified_date_range_answers_pass(simulator, answer):
    assert check_date_range_simplified(answer)


@pytest.mark.parametrize(
    "answer",
    [
        "Each range: OS101 and OS102, or UA200 and UA201.",
        "Each range: ZZ500 and ZZ600.",
        "Each range: 500 and 600.",
    ],
)
def test_simplified_date_range_lists_and_bare_numbers_fail(simulator, answer):
    assert not check_date_range_simplified(answer)
