from copy import deepcopy
from itertools import permutations, product
from unittest.mock import AsyncMock

import pytest

from westworld.base import UserMetadata, instantiate
from westworld.travelpedia.hotel_search import (
    create_application_state_from_dict,
    create_hotel_search_params_from_dict,
    verify_hotel_search_complete,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def room(age: int | None = None) -> dict:
    """Keep the age constraint optional to exercise the supported wildcard case."""
    result = {"adults": 1, "children": 1}
    if age is not None:
        result["child_ages"] = [age]
    return result


def search_state(rooms: list[dict]) -> dict:
    return {
        "destination": "Toronto, Canada",
        "checkin": "2027-01-10",
        "checkout": "2027-01-12",
        "roomDetails": rooms,
        "isSubmitted": True,
        "addFlight": False,
        "addCar": False,
    }


def search_values(rooms: list[dict]) -> dict:
    return {
        "city": "Toronto",
        "checkin": "2027-01-10",
        "checkout": "2027-01-12",
        "rooms": rooms,
        "addFlight": False,
        "addCar": False,
    }


def evaluator(values: dict):
    return instantiate({
        "_target_": "westworld.travelpedia.hotel_search.HalluminateTravelpediaHotelSearch",
        "task": "Submit a Toronto hotel search for the specified rooms.",
        "values": values,
        "user_metadata": UserMetadata(timestamp=0),
    })

# ---------------------------------------------------------------------------
# End fixture helpers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("expected", list(permutations([room(), room(5)])))
@pytest.mark.parametrize("actual", list(permutations([room(5), room(8)])))
async def test_valid_multi_room_score_is_independent_of_room_order(expected, actual):
    values = search_values(list(expected))
    state = search_state(list(actual))
    original_values, original_state = deepcopy(values), deepcopy(state)
    metric = evaluator(values)
    page = AsyncMock()
    page.evaluate.return_value = state

    await metric.update(page=page)
    result = await metric.compute()

    assert result.score == 1.0, result.reasoning
    assert values == original_values
    assert state == original_state
    page.evaluate.assert_awaited_once_with("() => window.__testingAPI.searchForm.getState()")


@pytest.mark.asyncio
@pytest.mark.parametrize("change", [
    {"destination": "Ottawa"},
    {"checkin": "2027-01-11"},
    {"checkout": "2027-01-13"},
    {"addFlight": True},
    {"addCar": True},
    {"isSubmitted": False},
    {"roomDetails": [room(5)]},
    {"roomDetails": [room(8), room(8)]},
    {"roomDetails": [room(5), {"adults": 2, "children": 1, "child_ages": [8]}]},
    {"roomDetails": [room(5), {"adults": 1, "children": 2, "child_ages": [8, 9]}]},
])
async def test_room_matching_preserves_other_rejections(change):
    metric = evaluator(search_values([room(), room(5)]))
    state = search_state([room(5), room(8)])
    state.update(change)
    page = AsyncMock()
    page.evaluate.return_value = state
    await metric.update(page=page)
    assert (await metric.compute()).score == 0.0


@pytest.mark.asyncio
async def test_duplicate_constraints_cannot_reuse_one_actual_room():
    metric = evaluator(search_values([room(5), room(5)]))
    page = AsyncMock()
    page.evaluate.return_value = search_state([room(5), room(8)])
    await metric.update(page=page)
    assert (await metric.compute()).score == 0.0


@pytest.mark.asyncio
async def test_no_page_empty_state_and_reset_still_fail():
    metric = evaluator(search_values([room(5)]))
    assert (await metric.compute()).score == 0.0
    page = AsyncMock()
    page.evaluate.return_value = {}
    await metric.update(page=page)
    assert (await metric.compute()).score == 0.0
    await metric.reset()
    assert (await metric.compute()).score == 0.0


def test_small_room_assignments_agree_with_exhaustive_bijection_oracle():
    """Check completeness and one-to-one use independently of the matching order."""
    cases = 0
    for count in range(1, 4):
        for expected in product([room(), room(5), room(8)], repeat=count):
            for actual in product([room(5), room(8)], repeat=count):
                possible = any(
                    all(
                        not wanted.get("child_ages") or wanted["child_ages"] == found["child_ages"]
                        for wanted, found in zip(expected, ordering)
                    )
                    for ordering in permutations(actual)
                )
                result = verify_hotel_search_complete(
                    create_application_state_from_dict(search_state(list(actual))),
                    create_hotel_search_params_from_dict(search_values(list(expected))),
                )
                assert result.passed == possible, (expected, actual, result.details)
                cases += 1
    assert cases == 258


@pytest.mark.parametrize("wildcard", [{}, {"child_ages": []}, {"child_ages": None}])
def test_existing_unspecified_age_semantics_are_preserved(wildcard):
    expected = room()
    expected.update(wildcard)
    result = verify_hotel_search_complete(
        create_application_state_from_dict(search_state([room(8)])),
        create_hotel_search_params_from_dict(search_values([expected])),
    )
    assert result.passed


def test_legacy_aggregated_search_still_passes():
    result = verify_hotel_search_complete(
        create_application_state_from_dict({
            "destination": "Toronto", "adults": 1, "children": 0,
            "child_ages": [], "isSubmitted": True,
        }),
        create_hotel_search_params_from_dict({
            "city": "Toronto", "adults": 1, "children": 0, "child_ages": [],
        }),
    )
    assert result.passed
