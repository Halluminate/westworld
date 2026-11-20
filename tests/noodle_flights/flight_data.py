"""Small Noodle Flights API payloads for verifier tests."""

from __future__ import annotations

from typing import Any


CABINS = ("ECONOMY", "PREMIUM_ECONOMY", "BUSINESS", "FIRST")


def flight(
    number: str,
    price: int,
    *,
    depart: str = "2030-06-12T08:00:00-07:00",
    arrive: str = "2030-06-12T16:30:00-04:00",
    seats: int = 9,
    origin: str = "SFO",
    destination: str = "JFK",
) -> dict[str, Any]:
    """A flight in the shape returned by /api/flights; connecting flights use "A/B" numbers."""
    legs = number.split("/")
    data: dict[str, Any] = {
        "flightNumber": number,
        "airline": legs[0][:2],
        "airlineName": legs[0][:2],
        "aircraft": "A320",
        "depart": depart,
        "arrive": arrive,
        "durationMin": 510,
        "distanceKm": 4100,
        "cabinPricing": [{"cabin": cabin, "priceUsd": price, "seatsRemaining": seats} for cabin in CABINS],
        "emissions": 320,
        "priceUsd": price,
        "originAirport": origin,
        "destinationAirport": destination,
    }
    if len(legs) > 1:
        data["segments"] = [
            {
                "flightNumber": leg,
                "airline": leg[:2],
                "airlineName": leg[:2],
                "aircraft": "A320",
                "depart": depart,
                "arrive": arrive,
                "durationMin": 240,
                "origin": origin if index == 0 else "ORD",
                "destination": destination if index == len(legs) - 1 else "ORD",
            }
            for index, leg in enumerate(legs)
        ]
    return data


def airport(iata: str) -> dict[str, Any]:
    return {
        "name": f"{iata} International Airport",
        "city": iata,
        "country": "Testland",
        "iata": iata,
        "lat": 0.0,
        "lon": 0.0,
        "tz": "UTC",
        "routeCount": 100,
        "airlines": {},
        "hubScore": 1,
    }


def airline(code: str) -> dict[str, Any]:
    return {"code": code, "name": code, "country": "Testland", "route_count": 10, "importance": 0.5, "hubs": []}
