import pytest

from errandworld import recipes
from errandworld.recipes import (
    BASE_ITEM_TYPES,
    CRAFTED_TYPES,
    RECIPES,
    STATION_TYPES,
    crafted_input,
    find_recipe,
    recipe_for,
    tier,
)


def test_each_station_type_has_tier1_and_tier2_recipes():
    for st in STATION_TYPES:
        tiers = sorted(tier(r.output) for r in RECIPES.values() if r.station_type == st)
        assert tiers == [1, 1, 2], st


def test_base_and_crafted_types_are_disjoint():
    assert not set(BASE_ITEM_TYPES) & CRAFTED_TYPES
    assert BASE_ITEM_TYPES == tuple(sorted(BASE_ITEM_TYPES))


def test_find_recipe_is_order_insensitive():
    assert find_recipe("repair_bench", ["bulb", "lamp_base"]).output == "lamp"
    assert find_recipe("repair_bench", ("lamp_base", "bulb")).output == "lamp"


def test_find_recipe_rejects_wrong_station_or_inputs():
    assert find_recipe("oven", ["bulb", "lamp_base"]) is None
    assert find_recipe("repair_bench", ["bulb"]) is None
    assert find_recipe("repair_bench", ["bulb", "bulb"]) is None


def test_tier2_recipes_consume_a_tier1_output():
    for out, r in RECIPES.items():
        if tier(out) == 2:
            ci = crafted_input(out)
            assert ci is not None and tier(ci) == 1
            assert recipe_for(ci).station_type in STATION_TYPES
        else:
            assert crafted_input(out) is None


def test_recipe_for_unknown_output():
    with pytest.raises(KeyError):
        recipe_for("spaceship")


def test_recipe_book_self_check_passes():
    recipes._check_book()
