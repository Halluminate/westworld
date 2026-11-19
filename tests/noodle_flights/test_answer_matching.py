from decimal import Decimal

from westworld.noodle_flights.answer_matching import (
    amount_mentioned,
    extract_amounts,
    extract_money_amounts,
    mentioned_flight_numbers,
    money_amount_mentioned,
    other_money_amounts,
)


def test_amounts_are_compared_as_numbers():
    assert amount_mentioned("The fare is $247.00.", 247)
    assert amount_mentioned("The fare is $247, nonstop.", 247)
    assert amount_mentioned("USD 1,247", 1247)
    assert amount_mentioned("1247 dollars", 1247)
    assert not amount_mentioned("$2,470", 247)
    assert not amount_mentioned("$1,247", 247)
    assert not amount_mentioned("Departs 2:47 PM", 247)
    assert not amount_mentioned("anything", None)


def test_extract_amounts_keeps_thousands_and_decimals_together():
    assert extract_amounts("$1,247.50 or 268, on 2030-06-12") == [
        Decimal("1247.50"),
        Decimal("268"),
        Decimal("2030"),
        Decimal("06"),
        Decimal("12"),
    ]


def test_money_amounts_need_a_currency_marker():
    text = "Fares: $1,234, 500 USD, USD 75 and 20 dollars, departing in 2030."
    assert extract_money_amounts(text) == [Decimal("1234"), Decimal("500"), Decimal("75"), Decimal("20")]
    assert money_amount_mentioned(text, 500)
    assert not money_amount_mentioned(text, 2030)


def test_flight_numbers_match_whole_tokens():
    candidates = ["UA389", "UA3891", "DL565", "AA108/AA985", "AA108/UA207"]
    assert mentioned_flight_numbers("Book UA389.", candidates) == {"UA389"}
    assert mentioned_flight_numbers("Book UA3891.", candidates) == {"UA3891"}
    assert mentioned_flight_numbers("Book UA 389 or DL-565", candidates) == {"UA389", "DL565"}
    assert mentioned_flight_numbers("book ua389", candidates) == set()


def test_connecting_itineraries_need_every_leg_close_together():
    candidates = ["AA108/AA985", "AA108/UA207", "AA985"]
    assert mentioned_flight_numbers("AA108/AA985 at $247", candidates) == {"AA108/AA985"}
    assert mentioned_flight_numbers("AA108 connecting to AA985", candidates) == {"AA108/AA985"}
    far_apart = "AA108 leaves early. " + "x" * 200 + " AA985 is a different itinerary."
    assert mentioned_flight_numbers(far_apart, candidates) == {"AA985"}


def test_other_money_amounts_ignores_allowed_values():
    text = "Outbound $500, return $600, total $2,200; another option was $520."
    assert other_money_amounts(text, [500, 520, 600, 640], [500, 600, 2200]) == [Decimal("520")]
    assert other_money_amounts("Total $2,200.", [500, 520], [2200]) == []
