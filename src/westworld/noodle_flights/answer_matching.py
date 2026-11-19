"""Matching an agent's free-text answer against Noodle Flights ground truth.

Prices are compared as numbers, so "$1,234", "1234", "$1,234.00" and "USD 1,234"
all match 1234. Flight numbers are matched as whole tokens, with an optional space
or hyphen between the airline code and the number ("AA108", "AA 108"), so "AA108"
does not match inside "AA1080". A connecting itinerary such as "AA108/AA985"
counts as mentioned when all of its legs are mentioned close together.
"""

from __future__ import annotations

import itertools
import re
from decimal import Decimal
from typing import Iterable


_NUMBER = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_NUMBER_END = r"(?!\d|,\d|\.\d)"
_AMOUNT_RE = re.compile(rf"(?<![\d.]){_NUMBER}{_NUMBER_END}")
_MONEY_RE = re.compile(
    rf"\$\s?(?P<prefixed>{_NUMBER}){_NUMBER_END}"
    rf"|\bUSD\s?(?P<code_first>{_NUMBER}){_NUMBER_END}"
    rf"|(?<![\d.])(?P<code_last>{_NUMBER})\s?(?:USD|dollars?)\b",
    re.IGNORECASE,
)
_LEG_RE = re.compile(r"^([A-Z0-9]{2})(\d{1,4})$")

# Legs of a connecting itinerary must appear within this many characters of each other.
LEG_WINDOW_CHARS = 80


def _to_decimal(raw: str) -> Decimal:
    return Decimal(raw.replace(",", ""))


def extract_amounts(text: str | None) -> list[Decimal]:
    """Every number in the text, with thousands separators removed."""
    return [_to_decimal(match.group(0)) for match in _AMOUNT_RE.finditer(text or "")]


def extract_money_amounts(text: str | None) -> list[Decimal]:
    """Numbers written as money: "$1,234", "USD 1234", "1,234 USD" or "1234 dollars"."""
    amounts = []
    for match in _MONEY_RE.finditer(text or ""):
        raw = match.group("prefixed") or match.group("code_first") or match.group("code_last")
        amounts.append(_to_decimal(raw))
    return amounts


def amount_mentioned(text: str | None, amount: int | float | Decimal | None) -> bool:
    """True when any number in the text equals the amount ("$247.00" matches 247)."""
    if amount is None:
        return False
    return Decimal(str(amount)) in extract_amounts(text)


def money_amount_mentioned(text: str | None, amount: int | float | Decimal | None) -> bool:
    """True when the amount appears in the text written as money."""
    if amount is None:
        return False
    return Decimal(str(amount)) in extract_money_amounts(text)


def other_money_amounts(
    text: str | None,
    candidate_amounts: Iterable[int | float | Decimal | None],
    allowed_amounts: Iterable[int | float | Decimal | None],
) -> list[Decimal]:
    """Money amounts in the text that equal a candidate amount but none of the allowed ones.

    Used to spot answers that list the prices of other options instead of committing
    to the requested one.
    """
    allowed = {Decimal(str(amount)) for amount in allowed_amounts if amount is not None}
    forbidden = {Decimal(str(amount)) for amount in candidate_amounts if amount is not None} - allowed
    found: list[Decimal] = []
    for amount in extract_money_amounts(text):
        if amount in forbidden and amount not in found:
            found.append(amount)
    return found


def _leg_spans(text: str, leg: str) -> list[tuple[int, int]]:
    match = _LEG_RE.match(leg.strip())
    if not match:
        return []
    code, number = match.groups()
    pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(code)}[\s-]?{number}(?!\d)")
    return [found.span() for found in pattern.finditer(text)]


def _itinerary_spans(text: str, flight_number: str) -> list[tuple[int, int]] | None:
    """Spans of the closest set of leg mentions for an itinerary, or None if not mentioned."""
    legs = [leg for leg in flight_number.split("/") if leg.strip()]
    spans_per_leg = [_leg_spans(text, leg) for leg in legs]
    if not legs or not all(spans_per_leg):
        return None
    best: list[tuple[int, int]] | None = None
    best_width = None
    for combination in itertools.product(*spans_per_leg):
        width = max(end for _, end in combination) - min(start for start, _ in combination)
        if width <= LEG_WINDOW_CHARS * max(len(legs) - 1, 1) and (best_width is None or width < best_width):
            best, best_width = list(combination), width
    return best


def mentioned_flight_numbers(text: str | None, flight_numbers: Iterable[str]) -> set[str]:
    """The flight numbers from ``flight_numbers`` that the text mentions.

    Connecting itineraries are resolved first; a single-leg flight number that only
    appears as a leg of a mentioned itinerary is not counted as a separate mention.
    """
    text = text or ""
    candidates = sorted({number for number in flight_numbers if number})
    mentioned: set[str] = set()
    consumed: set[tuple[int, int]] = set()
    for number in (n for n in candidates if "/" in n):
        spans = _itinerary_spans(text, number)
        if spans:
            mentioned.add(number)
            consumed.update(spans)
    for number in (n for n in candidates if "/" not in n):
        if any(span not in consumed for span in _leg_spans(text, number)):
            mentioned.add(number)
    return mentioned
