"""World transitions, error codes, observers, map view, and harness mutators.

Tiny map (conftest):  home — market — workshop
                        |        |
                      park — office
Items: battery_1, lamp_base_1 @ market; bulb_1 @ workshop; ink_1 @ park.
Stations: repair_bench_1 @ workshop, repair_bench_2 @ park. Capacity 3. Start: home.
"""

import pytest

from conftest import make_tiny_scenario
from errandworld.scenario_io import ScenarioError
from errandworld.types import AGENT, CONSUMED, Item
from errandworld.world import WORLD_ERROR_CODES, ErrorCode, World


@pytest.fixture
def world(tiny_scenario) -> World:
    return World.from_scenario(tiny_scenario)


def failed(world: World, action, *args, code: ErrorCode):
    """Run an action that must fail with ``code`` and leave the state untouched."""
    before = world.snapshot()
    outcome = action(*args)
    assert not outcome.ok
    assert outcome.error is code
    assert world.snapshot() == before
    return outcome


# ---------------------------------------------------------------- construction


def test_from_scenario_initial_state(world):
    assert world.agent_location == "home"
    assert world.inventory == []
    assert world.capacity == 3
    assert world.items_at("market") == ["battery_1", "lamp_base_1"]
    assert world.neighbors("market") == ["home", "office", "workshop"]
    assert world.stale_overrides == {}


# ---------------------------------------------------------------- move


def test_move_success(world):
    o = world.move("market")
    assert o.ok and world.agent_location == "market"
    assert o.data["previous"] == "home"
    assert [a["location"] for a in o.data["adjacent"]] == ["home", "office", "workshop"]


def test_move_unknown_location(world):
    failed(world, world.move, "mars", code=ErrorCode.UNKNOWN_ID)


def test_move_not_adjacent(world):
    failed(world, world.move, "workshop", code=ErrorCode.NOT_ADJACENT)


def test_move_to_current_location_is_not_adjacent(world):
    failed(world, world.move, "home", code=ErrorCode.NOT_ADJACENT)


def test_move_road_closed(world):
    world.close_road("market", "home")  # argument order does not matter
    failed(world, world.move, "market", code=ErrorCode.ROAD_CLOSED)
    assert world.open_neighbors("home") == ["park"]
    world.open_road("home", "market")
    assert world.move("market").ok


# ---------------------------------------------------------------- pick / drop


def test_pick_and_drop(world):
    world.move("market")
    o = world.pick("battery_1")
    assert o.ok and o.data["inventory"] == ["battery_1"]
    assert world.item_location["battery_1"] == AGENT
    assert world.items_at("market") == ["lamp_base_1"]
    world.move("office")
    o = world.drop("battery_1")
    assert o.ok and o.data["location"] == "office"
    assert world.item_location["battery_1"] == "office"
    assert world.inventory == []


def test_pick_unknown_item(world):
    failed(world, world.pick, "unicorn", code=ErrorCode.UNKNOWN_ID)


def test_pick_item_elsewhere(world):
    failed(world, world.pick, "battery_1", code=ErrorCode.ITEM_NOT_HERE)


def test_pick_item_already_carried(world):
    world.move("market")
    world.pick("battery_1")
    failed(world, world.pick, "battery_1", code=ErrorCode.ITEM_NOT_HERE)


def test_pick_inventory_full():
    items = tuple(Item(f"paper_{n}", "paper", "home") for n in range(1, 5))
    w = World.from_scenario(make_tiny_scenario(items=items + make_tiny_scenario().items))
    for n in range(1, 4):
        assert w.pick(f"paper_{n}").ok
    failed(w, w.pick, "paper_4", code=ErrorCode.INVENTORY_FULL)


def test_from_scenario_rejects_invalid_scenario():
    dup = (Item("ink_1", "ink", "home"),) + make_tiny_scenario().items  # ink_1 twice
    with pytest.raises(ScenarioError, match="duplicate item id"):
        World.from_scenario(make_tiny_scenario(items=dup))


def test_drop_unknown_item(world):
    failed(world, world.drop, "unicorn", code=ErrorCode.UNKNOWN_ID)


def test_drop_not_carrying(world):
    failed(world, world.drop, "battery_1", code=ErrorCode.NOT_CARRYING)


# ---------------------------------------------------------------- use


def carry_lamp_parts_to_workshop(w: World) -> None:
    assert w.move("market").ok
    assert w.pick("lamp_base_1").ok
    assert w.move("workshop").ok
    assert w.pick("bulb_1").ok


def test_use_crafts_output(world):
    carry_lamp_parts_to_workshop(world)
    o = world.use("repair_bench_1", ["bulb_1", "lamp_base_1"])  # order-insensitive
    assert o.ok
    assert o.data["output"] == "lamp_1" and o.data["output_type"] == "lamp"
    assert world.inventory == ["lamp_1"]
    assert world.item_location["lamp_base_1"] == CONSUMED
    assert world.item_location["bulb_1"] == CONSUMED
    assert world.item_type["lamp_1"] == "lamp"


def test_use_output_ids_are_unique_and_deterministic():
    extra = (
        Item("lamp_base_2", "lamp_base", "workshop"),
        Item("bulb_2", "bulb", "workshop"),
        Item("lamp_1", "lamp", "office"),  # pre-existing id must not be reused
    )
    w = World.from_scenario(make_tiny_scenario(items=make_tiny_scenario().items + extra))
    carry_lamp_parts_to_workshop(w)
    assert w.use("repair_bench_1", ["lamp_base_1", "bulb_1"]).data["output"] == "lamp_2"
    w.pick("lamp_base_2")
    w.pick("bulb_2")
    assert w.use("repair_bench_1", ["lamp_base_2", "bulb_2"]).data["output"] == "lamp_3"


def test_use_unknown_station(world):
    failed(world, world.use, "toaster_9", ["bulb_1"], code=ErrorCode.UNKNOWN_ID)


def test_use_unknown_item(world):
    carry_lamp_parts_to_workshop(world)
    failed(world, world.use, "repair_bench_1", ["bulb_1", "ghost"], code=ErrorCode.UNKNOWN_ID)


def test_use_not_at_station(world):
    carry_lamp_parts_to_workshop(world)
    failed(
        world, world.use, "repair_bench_2", ["bulb_1", "lamp_base_1"],
        code=ErrorCode.NOT_AT_STATION,
    )


def test_use_station_offline(world):
    carry_lamp_parts_to_workshop(world)
    world.set_station_online("repair_bench_1", False)
    failed(
        world, world.use, "repair_bench_1", ["bulb_1", "lamp_base_1"],
        code=ErrorCode.STATION_OFFLINE,
    )


def test_use_missing_items(world):
    world.move("market")
    world.pick("lamp_base_1")
    world.move("workshop")
    o = failed(
        world, world.use, "repair_bench_1", ["bulb_1", "lamp_base_1"],
        code=ErrorCode.MISSING_ITEMS,
    )
    assert o.data["items"] == ["bulb_1"]


def test_use_invalid_recipe(world):
    world.move("market")
    world.pick("battery_1")
    world.pick("lamp_base_1")
    world.move("workshop")
    failed(
        world, world.use, "repair_bench_1", ["battery_1", "lamp_base_1"],
        code=ErrorCode.INVALID_RECIPE,
    )


def test_use_duplicate_item_is_invalid_recipe(world):
    carry_lamp_parts_to_workshop(world)
    failed(world, world.use, "repair_bench_1", ["bulb_1", "bulb_1"], code=ErrorCode.INVALID_RECIPE)


def test_every_error_code_is_exercised():
    """Guard: adding an ErrorCode without a test above should fail loudly."""
    tested = {
        ErrorCode.NOT_ADJACENT, ErrorCode.ROAD_CLOSED, ErrorCode.ITEM_NOT_HERE,
        ErrorCode.INVENTORY_FULL, ErrorCode.NOT_CARRYING, ErrorCode.NOT_AT_STATION,
        ErrorCode.STATION_OFFLINE, ErrorCode.MISSING_ITEMS, ErrorCode.INVALID_RECIPE,
        ErrorCode.UNKNOWN_ID,
    }
    assert tested == set(WORLD_ERROR_CODES)


# ---------------------------------------------------------------- observers


def test_look_shows_true_local_state(world):
    world.move("market")
    world.move("workshop")
    world.set_station_online("repair_bench_1", False)
    world.close_road("workshop", "market")
    v = world.look()
    assert v["location"] == "workshop"
    assert v["items"] == [{"id": "bulb_1", "type": "bulb"}]
    assert v["stations"] == [{"id": "repair_bench_1", "type": "repair_bench", "online": False}]
    assert v["roads"] == [{"to": "market", "open": False}]


def test_status_reports_inventory(world):
    world.move("market")
    world.pick("battery_1")
    assert world.status() == {
        "location": "market",
        "inventory": [{"id": "battery_1", "type": "battery"}],
        "capacity": 3,
    }


def test_read_map_omits_carried_and_consumed_items(world):
    carry_lamp_parts_to_workshop(world)
    world.use("repair_bench_1", ["lamp_base_1", "bulb_1"])
    ids = [i["id"] for i in world.read_map()["items"]]
    assert ids == ["battery_1", "ink_1"]
    world.drop("lamp_1")
    assert {"id": "lamp_1", "type": "lamp", "location": "workshop"} in world.read_map()["items"]


# ---------------------------------------------------------------- map view (D5)


def map_location(world: World, item: str) -> str | None:
    for i in world.read_map()["items"]:
        if i["id"] == item:
            return i["location"]
    return None


def test_synced_changes_appear_on_map(world):
    world.close_road("home", "market")
    world.set_station_online("repair_bench_2", False)
    world.move_item("battery_1", "park", update_map=True)
    m = world.read_map()
    assert {"a": "home", "b": "market", "open": False} in m["roads"]
    assert next(s for s in m["stations"] if s["id"] == "repair_bench_2")["online"] is False
    assert map_location(world, "battery_1") == "park"


def test_stale_move_keeps_old_map_entry_but_look_shows_truth(world):
    world.move_item("battery_1", "park", update_map=False)
    assert map_location(world, "battery_1") == "market"  # map is stale
    assert world.item_location["battery_1"] == "park"  # truth
    world.move("market")
    assert "battery_1" not in [i["id"] for i in world.look()["items"]]
    assert world.pick("battery_1").error is ErrorCode.ITEM_NOT_HERE


def test_repeated_stale_moves_keep_original_belief(world):
    world.move_item("battery_1", "park", update_map=False)
    world.move_item("battery_1", "office", update_map=False)
    assert map_location(world, "battery_1") == "market"


def test_picking_stale_item_clears_override(world):
    world.move_item("battery_1", "park", update_map=False)
    world.move("park")
    assert world.pick("battery_1").ok
    assert world.stale_overrides == {}
    world.drop("battery_1")
    assert map_location(world, "battery_1") == "park"


def test_synced_move_clears_override(world):
    world.move_item("battery_1", "park", update_map=False)
    world.move_item("battery_1", "office", update_map=True)
    assert map_location(world, "battery_1") == "office"


# ---------------------------------------------------------------- harness mutators


def test_harness_mutators_raise_on_misuse(world):
    with pytest.raises(KeyError):
        world.close_road("home", "workshop")  # no such road
    with pytest.raises(KeyError):
        world.set_station_online("ghost", False)
    with pytest.raises(KeyError):
        world.move_item("ghost", "home")
    with pytest.raises(KeyError):
        world.move_item("battery_1", "mars")
    world.move("market")
    world.pick("battery_1")
    with pytest.raises(ValueError):
        world.move_item("battery_1", "home")  # carried items cannot be relocated


def test_clone_is_independent(world):
    c = world.clone()
    c.move("market")
    c.pick("battery_1")
    c.close_road("home", "park")
    c.set_station_online("repair_bench_1", False)
    c.move_item("ink_1", "office", update_map=False)
    assert world.agent_location == "home"
    assert world.inventory == []
    assert world.is_road_open("home", "park")
    assert world.station_online["repair_bench_1"]
    assert world.stale_overrides == {}
    assert world.snapshot() == World.from_scenario(make_tiny_scenario()).snapshot()


def test_snapshot_is_deterministic(world):
    a = world.clone()
    b = world.clone()
    for w in (a, b):
        carry_lamp_parts_to_workshop(w)
        w.use("repair_bench_1", ["lamp_base_1", "bulb_1"])
    assert a.snapshot() == b.snapshot()
