"""The fixed recipe book shared by every ErrandWorld map and scenario.

Three station types, three recipes each. Every recipe takes exactly two inputs.

- Tier 1 recipes use only base items (items that can be placed on the map).
- Tier 2 recipes take one tier-1 output as an input. A craft job for a tier-2 recipe is a
  *dependent* job: it consumes the item delivered by the tier-1 job it depends on.

Recipe outputs are unique, so a recipe is identified by its output type.
"""

from __future__ import annotations

from .types import Recipe

STATION_TYPES: tuple[str, ...] = ("oven", "print_shop", "repair_bench")

_RECIPE_LIST: tuple[Recipe, ...] = (
    # repair_bench
    Recipe("repair_bench", ("lamp_base", "bulb"), "lamp"),
    Recipe("repair_bench", ("radio_shell", "battery"), "radio"),
    Recipe("repair_bench", ("radio", "antenna"), "tuned_radio"),  # tier 2
    # oven
    Recipe("oven", ("flour", "water"), "bread"),
    Recipe("oven", ("sugar", "eggs"), "cake"),
    Recipe("oven", ("bread", "cheese"), "toasted_sandwich"),  # tier 2
    # print_shop
    Recipe("print_shop", ("paper", "ink"), "poster"),
    Recipe("print_shop", ("card", "stamp"), "postcard"),
    Recipe("print_shop", ("poster", "frame"), "framed_poster"),  # tier 2
)

RECIPES: dict[str, Recipe] = {r.output: r for r in _RECIPE_LIST}

CRAFTED_TYPES: frozenset[str] = frozenset(RECIPES)

BASE_ITEM_TYPES: tuple[str, ...] = tuple(
    sorted({t for r in _RECIPE_LIST for t in r.inputs} - CRAFTED_TYPES)
)


def recipe_for(output_type: str) -> Recipe:
    try:
        return RECIPES[output_type]
    except KeyError:
        raise KeyError(f"no recipe produces {output_type!r}") from None


def find_recipe(station_type: str, input_types: list[str] | tuple[str, ...]) -> Recipe | None:
    """The recipe a station of ``station_type`` would run on these inputs, or None."""
    for output in sorted(RECIPES):
        r = RECIPES[output]
        if r.matches(station_type, input_types):
            return r
    return None


def tier(output_type: str) -> int:
    """1 if the recipe uses only base items, 2 if it consumes another recipe's output."""
    r = recipe_for(output_type)
    return 2 if any(t in CRAFTED_TYPES for t in r.inputs) else 1


def crafted_input(output_type: str) -> str | None:
    """For a tier-2 recipe, the tier-1 output it consumes; otherwise None."""
    r = recipe_for(output_type)
    for t in r.inputs:
        if t in CRAFTED_TYPES:
            return t
    return None


def _check_book() -> None:
    assert len(RECIPES) == len(_RECIPE_LIST), "recipe outputs must be unique"
    for r in _RECIPE_LIST:
        assert r.station_type in STATION_TYPES, r
        assert len(r.inputs) == 2, r
        assert r.output not in BASE_ITEM_TYPES, r
        assert sum(t in CRAFTED_TYPES for t in r.inputs) <= 1, r
    for st in STATION_TYPES:
        assert any(r.station_type == st for r in _RECIPE_LIST), st


_check_book()
