"""Tool layer: specs, argument validation, dispatch, rendered texts, and resources.

Rendered texts are what the LLM reads, so they are pinned exactly: any change to them is a change
to the experiment and should show up as a failing test.
"""

import json

import pytest

from conftest import make_tiny_scenario
from errandworld.tools import (
    READ_TOOLS,
    TOOL_SPECS,
    WRITE_TOOLS,
    ToolKind,
    dispatch,
    function_definitions,
)
from errandworld.world import ErrorCode, World


@pytest.fixture
def world(tiny_scenario) -> World:
    return World.from_scenario(tiny_scenario)


def call(world, name, **args):
    return dispatch(world, name, args)


# ---------------------------------------------------------------- specs


def test_seven_tools_with_fixed_read_write_split():
    assert list(TOOL_SPECS) == ["move", "pick", "drop", "use", "look", "status", "read_map"]
    assert WRITE_TOOLS == {"move", "pick", "drop", "use"}
    assert READ_TOOLS == {"look", "status", "read_map"}


def test_schemas_are_strict_objects():
    for spec in TOOL_SPECS.values():
        p = spec.parameters
        assert p["type"] == "object"
        assert p["additionalProperties"] is False
        assert sorted(p["required"]) == sorted(p["properties"])
    assert TOOL_SPECS["use"].required == ("items", "station")


def test_function_definitions_are_json_and_ordered():
    defs = function_definitions()
    assert [d["function"]["name"] for d in defs] == list(TOOL_SPECS)
    json.dumps(defs)  # serializable


def test_write_tool_descriptions_mention_action_cost():
    for name, spec in TOOL_SPECS.items():
        if spec.kind is ToolKind.WRITE:
            assert "Counts as one action" in spec.description, name
        else:
            assert "Does not count as an action" in spec.description, name


# ---------------------------------------------------------------- validation


@pytest.mark.parametrize(
    "name, args, fragment",
    [
        ("move", {}, "Missing argument(s) for move: location."),
        ("move", {"location": 3}, "'location' for move must be a string"),
        ("move", {"location": "market", "speed": "fast"}, "Unexpected argument(s) for move: speed"),
        ("look", {"verbose": True}, "Unexpected argument(s) for look: verbose. Allowed: none."),
        ("use", {"station": "repair_bench_1", "items": "bulb_1"}, "must be a list of strings"),
        ("use", {"station": "repair_bench_1", "items": []}, "must not be empty"),
        ("pick", ["battery_1"], "must be an object, got list"),
    ],
)
def test_invalid_arguments(world, name, args, fragment):
    before = world.snapshot()
    r = dispatch(world, name, args)
    assert not r.ok and r.error is ErrorCode.INVALID_ARGUMENTS
    assert fragment in r.text
    assert r.text.startswith("ERROR invalid_arguments: ")
    assert r.is_write == (name in WRITE_TOOLS)
    assert world.snapshot() == before


def test_unknown_tool(world):
    r = dispatch(world, "teleport", {"location": "office"})
    assert not r.ok and r.error is ErrorCode.UNKNOWN_TOOL
    assert r.text.startswith("ERROR unknown_tool: Unknown tool 'teleport'.")
    assert r.is_write is False


def test_none_args_allowed_for_read_tools(world):
    assert dispatch(world, "look", None).ok


def test_dispatch_never_raises_on_garbage(world):
    for name, args in [(None, None), (42, {}), ("use", {"station": None, "items": [1]})]:
        r = dispatch(world, name, args)  # type: ignore[arg-type]
        assert not r.ok


# ---------------------------------------------------------------- rendered texts (pinned)


def test_move_texts(world):
    assert call(world, "move", location="market").text == (
        "You moved to market. Roads from here: home (open), office (open), workshop (open)."
    )
    assert call(world, "move", location="park").text == (
        "ERROR not_adjacent: park is not adjacent to market. You are still at market."
    )
    assert call(world, "move", location="mars").text == (
        "ERROR unknown_id: There is no location called 'mars'."
    )
    world.close_road("market", "workshop")
    assert call(world, "move", location="workshop").text == (
        "ERROR road_closed: The road from market to workshop is closed. You are still at market."
    )


def test_pick_and_drop_texts(world):
    assert call(world, "pick", item="battery_1").text == (
        "ERROR item_not_here: battery_1 is not at home."
    )
    call(world, "move", location="market")
    assert call(world, "pick", item="battery_1").text == (
        "You picked up battery_1 (battery). Inventory (1/3): battery_1 (battery)."
    )
    assert call(world, "drop", item="ink_1").text == (
        "ERROR not_carrying: You are not carrying ink_1. Inventory (1/3): battery_1 (battery)."
    )
    assert call(world, "drop", item="battery_1").text == (
        "You dropped battery_1 (battery) at market. Inventory (0/3): empty."
    )
    assert call(world, "pick", item="ghost").text == "ERROR unknown_id: There is no item called 'ghost'."


def test_inventory_full_text():
    from errandworld.types import Item

    items = tuple(Item(f"paper_{n}", "paper", "home") for n in range(1, 5))
    w = World.from_scenario(make_tiny_scenario(items=items + make_tiny_scenario().items))
    for n in range(1, 4):
        call(w, "pick", item=f"paper_{n}")
    assert call(w, "pick", item="paper_4").text == (
        "ERROR inventory_full: Your inventory is full (3/3); drop something first. "
        "Inventory (3/3): paper_1 (paper), paper_2 (paper), paper_3 (paper)."
    )


def test_use_texts(world):
    call(world, "move", location="market")
    call(world, "pick", item="lamp_base_1")
    assert call(world, "use", station="repair_bench_1", items=["lamp_base_1"]).text == (
        "ERROR not_at_station: repair_bench_1 is not at market."
    )
    call(world, "move", location="workshop")
    assert call(world, "use", station="repair_bench_1", items=["lamp_base_1", "bulb_1"]).text == (
        "ERROR missing_items: You are not carrying bulb_1. Inventory (1/3): lamp_base_1 (lamp_base)."
    )
    call(world, "pick", item="bulb_1")
    assert call(world, "use", station="repair_bench_1", items=["bulb_1"]).text == (
        "ERROR invalid_recipe: repair_bench_1 (repair_bench) has no recipe for those items."
    )
    assert call(world, "use", station="oven_7", items=["bulb_1"]).text == (
        "ERROR unknown_id: There is no station called 'oven_7'."
    )
    assert call(world, "use", station="repair_bench_1", items=["bulb_1", "x"]).text == (
        "ERROR unknown_id: There is no item called 'x'."
    )
    world.set_station_online("repair_bench_1", False)
    assert call(world, "use", station="repair_bench_1", items=["bulb_1", "lamp_base_1"]).text == (
        "ERROR station_offline: repair_bench_1 is offline."
    )
    world.set_station_online("repair_bench_1", True)
    r = call(world, "use", station="repair_bench_1", items=["lamp_base_1", "bulb_1"])
    assert r.text == (
        "repair_bench_1 consumed lamp_base_1, bulb_1 and produced lamp_1 (lamp). "
        "Inventory (1/3): lamp_1 (lamp)."
    )


def test_look_text(world):
    call(world, "move", location="park")
    world.set_station_online("repair_bench_2", False)
    world.close_road("park", "office")
    assert call(world, "look").text == (
        "You are at park.\n"
        "Items here: ink_1 (ink).\n"
        "Stations here: repair_bench_2 (repair_bench, offline).\n"
        "Roads from here: home (open), office (closed)."
    )


def test_look_text_empty_location(world):
    call(world, "move", location="park")
    call(world, "move", location="office")
    assert call(world, "look").text == (
        "You are at office.\n"
        "Items here: none.\n"
        "Stations here: none.\n"
        "Roads from here: market (open), park (open)."
    )


def test_status_text_with_and_without_jobs(world):
    assert call(world, "status").text == "You are at home.\nInventory (0/3): empty."
    jobs = [
        {"id": "job_1", "description": "Deliver a battery to office.", "done": True},
        {"id": "job_2", "description": "Make a lamp and deliver it to home.", "done": False},
    ]
    r = dispatch(world, "status", {}, jobs=jobs)
    assert r.text == (
        "You are at home.\n"
        "Inventory (0/3): empty.\n"
        "Jobs:\n"
        "- job_1 [done]: Deliver a battery to office.\n"
        "- job_2 [pending]: Make a lamp and deliver it to home."
    )
    assert r.data["jobs"] == jobs


def test_read_map_text_shows_stale_entry(world):
    world.move_item("battery_1", "park", update_map=False)
    world.close_road("home", "park")
    assert call(world, "read_map").text == (
        "Locations: home, market, office, park, workshop.\n"
        "Roads:\n"
        "- home <-> market (open)\n"
        "- home <-> park (closed)\n"
        "- market <-> office (open)\n"
        "- market <-> workshop (open)\n"
        "- office <-> park (open)\n"
        "Stations:\n"
        "- repair_bench_1 (repair_bench) at workshop, online\n"
        "- repair_bench_2 (repair_bench) at park, online\n"
        "Items:\n"
        "- battery_1 (battery) at market\n"  # stale: truly at park
        "- bulb_1 (bulb) at workshop\n"
        "- ink_1 (ink) at park\n"
        "- lamp_base_1 (lamp_base) at market"
    )


def test_error_texts_always_carry_code_prefix(world):
    cases = [
        ("move", {"location": "workshop"}),
        ("pick", {"item": "bulb_1"}),
        ("drop", {"item": "bulb_1"}),
        ("use", {"station": "repair_bench_1", "items": ["bulb_1"]}),
    ]
    for name, args in cases:
        r = dispatch(world, name, args)
        assert not r.ok
        assert r.text.startswith(f"ERROR {r.error.value}: ")


# ---------------------------------------------------------------- result metadata


def test_results_record_resources(world):
    r = call(world, "move", location="market")
    assert r.resources == ("road:home|market", "location:market")
    assert call(world, "pick", item="lamp_base_1").resources == ("item:lamp_base_1",)
    failed_move = call(world, "move", location="park")  # not adjacent: no road touched
    assert failed_move.resources == ("location:park",)
    r = call(world, "use", station="repair_bench_1", items=["lamp_base_1", "bulb_1"])
    assert r.resources == ("station:repair_bench_1", "item:lamp_base_1", "item:bulb_1")
    assert call(world, "look").resources == ()


def test_is_write_flag(world):
    assert call(world, "move", location="market").is_write
    assert call(world, "move", location="park").is_write  # failed writes are still writes
    assert not call(world, "look").is_write


def test_to_dict_is_json_serializable(world):
    for r in [
        call(world, "move", location="market"),
        call(world, "move", location="park"),
        call(world, "read_map"),
        dispatch(world, "nope", None),
    ]:
        d = r.to_dict()
        json.dumps(d)
        assert d["tool"] == r.tool and d["ok"] == r.ok


def test_dispatch_is_deterministic(tiny_scenario):
    script = [
        ("move", {"location": "market"}), ("pick", {"item": "lamp_base_1"}),
        ("move", {"location": "workshop"}), ("pick", {"item": "bulb_1"}),
        ("use", {"station": "repair_bench_1", "items": ["lamp_base_1", "bulb_1"]}),
        ("look", {}), ("read_map", {}),
    ]
    runs = []
    for _ in range(2):
        w = World.from_scenario(tiny_scenario)
        runs.append([dispatch(w, n, a).to_dict() for n, a in script])
    assert runs[0] == runs[1]
