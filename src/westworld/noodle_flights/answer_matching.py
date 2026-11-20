"""Matching an agent's free-text answer against Noodle Flights ground truth.

Prices are compared as numbers. A price written as money counts: "$247", "$247.00",
"US$247", "USD 247", "247 USD", "247 dollars" or "247$". Where a task also accepts a
bare number ("247"), the number must stand on its own: digits inside a flight number
("OS2200"), a time ("22:00"), a date ("2030-06-12") or a year ("2030") never count.

Flight numbers are matched as whole tokens: "AA108", "AA 108" or "AA-108", and in any
case when written solid ("aa108"), but never inside a longer token such as "AA1080".
A connecting itinerary such as "AA108/AA985" counts as mentioned when every leg is
named. When a leg also belongs to another candidate itinerary or flight, the legs must
appear within SHARED_LEG_WINDOW characters of each other.
"""

from __future__ import annotations

import re
from collections import Counter
from decimal import Decimal
from typing import Iterable


_NUMBER = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_NUMBER_END = r"(?!\d|[.,]\d)"
_MONEY_RE = re.compile(
    rf"(?:US)?\$\s?(?P<dollar_first>{_NUMBER}){_NUMBER_END}"
    rf"|\bUSD\s?(?P<code_first>{_NUMBER}){_NUMBER_END}"
    rf"|(?<![A-Za-z0-9.,$])(?P<code_last>{_NUMBER})\s?(?:USD|dollars?)\b"
    rf"|(?<![A-Za-z0-9.,$])(?P<dollar_last>{_NUMBER})\$(?!\d)",
    re.IGNORECASE,
)
# A bare number: not glued to letters (flight numbers, "8h"), not part of a time or a date.
_BARE_RE = re.compile(rf"(?<![A-Za-z0-9.,:/$-]){_NUMBER}(?![A-Za-z]|\d|[.,:/-]\d)")
_LEG_RE = re.compile(r"^([A-Z]{2,3}|[A-Z]\d|\d[A-Z])(\d{1,4})$")

# Legs shared with other candidates must appear within this many characters of each other.
SHARED_LEG_WINDOW = 300
# Mentions looked at per leg; bounds the matching cost for long or repetitive responses.
MAX_SPANS_PER_LEG = 50


def _to_decimal(raw: str) -> Decimal:
    return Decimal(raw.replace(",", ""))


def extract_money_amounts(text: str | None) -> list[Decimal]:
    """Numbers written as money: "$1,234", "US$1234", "USD 1234", "1,234 USD", "1234 dollars", "1234$"."""
    amounts = []
    for match in _MONEY_RE.finditer(text or ""):
        raw = next(group for group in match.groups() if group)
        amounts.append(_to_decimal(raw))
    return amounts


def extract_bare_amounts(text: str | None) -> list[Decimal]:
    """Standalone numbers that are not part of a code, a time, a date, or a year (1900-2100 without a comma)."""
    amounts = []
    for match in _BARE_RE.finditer(text or ""):
        raw = match.group(0)
        if "," not in raw and "." not in raw and len(raw) == 4 and 1900 <= int(raw) <= 2100:
            continue
        amounts.append(_to_decimal(raw))
    return amounts


def price_mentioned(text: str | None, amount: int | float | Decimal | None, *, allow_bare: bool = False) -> bool:
    """True when the text states the amount as money, or, with allow_bare, as a standalone number."""
    if amount is None:
        return False
    value = Decimal(str(amount))
    if value in extract_money_amounts(text):
        return True
    return allow_bare and value in extract_bare_amounts(text)


def other_option_amounts(
    text: str | None,
    candidate_amounts: Iterable[int | float | Decimal | None],
    allowed_amounts: Iterable[int | float | Decimal | None],
) -> list[Decimal]:
    """Amounts in the text, with or without a currency marker, that equal a candidate but no allowed amount.

    Used to spot answers that quote the prices of other options instead of committing
    to the requested one.
    """
    allowed = {Decimal(str(amount)) for amount in allowed_amounts if amount is not None}
    forbidden = {Decimal(str(amount)) for amount in candidate_amounts if amount is not None} - allowed
    found: list[Decimal] = []
    for amount in [*extract_money_amounts(text), *extract_bare_amounts(text)]:
        if amount in forbidden and amount not in found:
            found.append(amount)
    return found


def _legs(flight_number: str) -> list[str]:
    return [leg.strip().upper() for leg in flight_number.split("/") if leg.strip()]


def _leg_pattern(leg: str) -> re.Pattern[str]:
    match = _LEG_RE.match(leg)
    if match:
        code, number = match.groups()
        # Upper-case code with an optional separator, or any case when written solid;
        # prose such as "to 12" or "as 2" does not match flight TO12 or AS2.
        body = rf"(?:{code}[\s-]?{number}|(?i:{code}){number})"
    else:
        body = rf"(?i:{re.escape(leg)})"
    return re.compile(rf"(?<![A-Za-z0-9]){body}(?!\d)")


def _leg_spans(text: str, leg: str) -> list[tuple[int, int]]:
    spans = []
    for found in _leg_pattern(leg).finditer(text):
        spans.append(found.span())
        if len(spans) == MAX_SPANS_PER_LEG:
            break
    return spans


def _spans_in_windows(spans_per_leg: list[list[tuple[int, int]]], window: int) -> set[tuple[int, int]]:
    """Spans that belong to a stretch of text, at most ``window`` characters wide, naming every leg."""
    events = sorted((start, end, leg) for leg, spans in enumerate(spans_per_leg) for start, end in spans)
    needed = len(spans_per_leg)
    counts = [0] * needed
    covered = 0
    left = 0
    used: set[tuple[int, int]] = set()
    for right, (start, _, leg) in enumerate(events):
        if counts[leg] == 0:
            covered += 1
        counts[leg] += 1
        while start - events[left][0] > window:
            counts[events[left][2]] -= 1
            if counts[events[left][2]] == 0:
                covered -= 1
            left += 1
        if covered == needed:
            used.update((s, e) for s, e, _ in events[left : right + 1])
    return used


def mentioned_flight_numbers(text: str | None, flight_numbers: Iterable[str]) -> set[str]:
    """The flight numbers from ``flight_numbers`` that the text mentions.

    Connecting itineraries are resolved first. A single-leg flight number counts only
    where it is mentioned outside the itineraries already matched, so naming
    "AA108/AA985" does not also count as naming a direct flight "AA108".
    """
    text = text or ""
    candidates = sorted({number.strip() for number in flight_numbers if number and number.strip()})
    legs_of = {number: _legs(number) for number in candidates}
    users = Counter(leg for legs in legs_of.values() for leg in set(legs))
    spans = {leg: _leg_spans(text, leg) for leg in users}

    mentioned: set[str] = set()
    consumed: set[tuple[int, int]] = set()
    for number, legs in legs_of.items():
        if len(legs) < 2 or not all(spans[leg] for leg in legs):
            continue
        if all(users[leg] == 1 for leg in legs):
            # No other candidate uses these legs, so they can be named anywhere in the text.
            used = {span for leg in legs for span in spans[leg]}
        else:
            used = _spans_in_windows([spans[leg] for leg in legs], SHARED_LEG_WINDOW)
        if used:
            mentioned.add(number)
            consumed.update(used)
    for number, legs in legs_of.items():
        if len(legs) == 1 and any(span not in consumed for span in spans[legs[0]]):
            mentioned.add(number)
    return mentioned
