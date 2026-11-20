import time
from decimal import Decimal

from westworld.noodle_flights.answer_matching import (
    extract_bare_amounts,
    extract_money_amounts,
    mentioned_flight_numbers,
    other_option_amounts,
    price_mentioned,
)


def test_money_amounts_in_common_formats():
    text = "Fares: $247, US$1,234.50, USD 75, 500 USD, 20 dollars and 99$."
    assert extract_money_amounts(text) == [
        Decimal("247"),
        Decimal("1234.50"),
        Decimal("75"),
        Decimal("500"),
        Decimal("20"),
        Decimal("99"),
    ]


def test_digits_inside_codes_times_dates_and_years_are_not_amounts():
    text = "OS2200 USD, gate B247, 8:52 PM, 8h 43m, 2030-06-12, 06/12/2030, 12.06.2030, June 12, 2030"
    assert extract_money_amounts(text) == []
    assert extract_bare_amounts(text) == [Decimal("12")]  # the day in "June 12"
    assert extract_bare_amounts("costs 247, or 2,030 in total") == [Decimal("247"), Decimal("2030")]


def test_price_needs_a_currency_marker_unless_bare_numbers_are_allowed():
    assert price_mentioned("The fare is $247.00.", 247)
    assert price_mentioned("247 dollars", 247)
    assert not price_mentioned("The fare is 247.", 247)
    assert price_mentioned("The fare is 247.", 247, allow_bare=True)
    assert not price_mentioned("$2,470", 247, allow_bare=True)
    assert not price_mentioned("anything", None)


def test_price_is_not_read_from_a_flight_number_a_time_or_a_year():
    assert not price_mentioned("Flight OS2200 costs $9,999", 2200)
    assert not price_mentioned("Flight OS2200 costs $9,999", 2200, allow_bare=True)
    assert not price_mentioned("ZZ500 and ZZ600", 500, allow_bare=True)
    assert not price_mentioned("Departs 20:30 on June 12, 2030", 2030, allow_bare=True)
    assert price_mentioned("It costs $2,030", 2030)


def test_other_option_amounts_count_prices_with_or_without_a_currency_marker():
    text = "OS101 500, UA200 520, OS102 600, UA201 640. Total $2,200"
    assert other_option_amounts(text, [500, 520, 600, 640], [500, 600, 2200]) == [Decimal("520"), Decimal("640")]
    assert other_option_amounts("Total $2,200 on June 12, 2030", [2030, 520], [2200]) == []


def test_flight_numbers_match_whole_tokens():
    candidates = ["UA389", "UA3891", "DL565", "TO12"]
    assert mentioned_flight_numbers("Book UA389.", candidates) == {"UA389"}
    assert mentioned_flight_numbers("Book UA3891.", candidates) == {"UA3891"}
    assert mentioned_flight_numbers("Book UA 389 or DL-565", candidates) == {"UA389", "DL565"}
    assert mentioned_flight_numbers("book ua389 or dl565", candidates) == {"UA389", "DL565"}
    assert mentioned_flight_numbers("we fly to 12 cities", candidates) == set()
    assert mentioned_flight_numbers("Take TO 12", candidates) == {"TO12"}


def test_three_letter_airline_codes_match():
    assert mentioned_flight_numbers("EZY123 or AAL108", ["EZY123", "AAL108", "AA108"]) == {"EZY123", "AAL108"}


def test_connecting_legs_can_be_far_apart_when_no_other_candidate_shares_them():
    answer = (
        "AA108 (SFO 8:00 AM -> ORD 2:05 PM, 4h 05m), 1h 20m layover in Chicago O'Hare, "
        "then AA985 (ORD 3:25 PM -> JFK 6:35 PM, 2h 10m). Price $247."
    )
    assert mentioned_flight_numbers(answer, ["AA108/AA985", "UA835/WS935", "UA389"]) == {"AA108/AA985"}


def test_shared_legs_must_appear_together():
    candidates = ["AA108/AA985", "AA108/UA207", "AA985"]
    assert mentioned_flight_numbers("AA108 connecting to AA985", candidates) == {"AA108/AA985"}
    far_apart = "AA108 leaves early. " + "x" * 400 + " AA985 is the evening nonstop."
    assert mentioned_flight_numbers(far_apart, candidates) == {"AA985"}


def test_itinerary_named_twice_is_not_also_its_leg():
    candidates = ["AA108/AA985", "AA985"]
    answer = "Book AA108/AA985. To confirm: AA108/AA985 is the cheapest."
    assert mentioned_flight_numbers(answer, candidates) == {"AA108/AA985"}


def test_repeated_mentions_stay_fast():
    text = " ".join(["XX1/XX2/XX3 connects"] * 150)
    started = time.perf_counter()
    mentioned = mentioned_flight_numbers(text, ["XX1/XX2/XX3", "XX9/XX2/XX3", "XX1"])
    assert mentioned == {"XX1/XX2/XX3"}
    assert time.perf_counter() - started < 0.5
