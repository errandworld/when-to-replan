import pytest

from app.checker import Beliefs, Trigger, contradiction, failure
from app.plan import PlanStep, Precondition
from app.types import ToolResult


def ok(tool, args=None, **payload):
    return ToolResult(tool, args or {}, ok=True, payload={"ok": True, **payload})


def failed(tool, args, error):
    return ToolResult(tool, args, ok=False, error=error, payload={"ok": False, "error": error})


def at(location):
    return Precondition("at", {"location": location})


def carrying(item):
    return Precondition("carrying", {"item": item})


def item_at(item, location):
    return Precondition("item_at", {"item": item, "location": location})


def road(a, b):
    return Precondition("road_open", {"from": a, "to": b})


def online(station):
    return Precondition("station_online", {"station": station})


def beliefs_after(*results):
    beliefs = Beliefs()
    for result in results:
        beliefs.update(result)
    return beliefs


@pytest.mark.parametrize(
    "precondition", [at("home"), carrying("bulb"), item_at("bulb", "shop"), road("a", "b"), online("repair_1")]
)
def test_nothing_is_known_at_first(precondition):
    assert Beliefs().holds(precondition) is None


def test_location_from_status_and_move():
    beliefs = beliefs_after(ok("status", location="home", inventory=[]))
    assert beliefs.location == "home" and beliefs.holds(at("home")) and beliefs.holds(at("shop")) is False

    beliefs.update(ok("move", {"location": "shop"}, location="shop", adjacent=["home"]))
    assert beliefs.location == "shop"
    assert beliefs.holds(road("shop", "home")) is True  # roads are two-way


def test_move_without_a_location_in_the_payload_still_moves():
    beliefs = beliefs_after(ok("status", location="home"), ok("move", {"location": "shop"}))
    assert beliefs.location == "shop"


@pytest.mark.parametrize("error", ["road_closed", "not_adjacent"])
def test_failed_move_closes_the_road_and_stays_put(error):
    beliefs = beliefs_after(ok("status", location="home"), failed("move", {"location": "shop"}, error))
    assert beliefs.location == "home"
    assert beliefs.holds(road("home", "shop")) is False


def test_failed_move_from_an_unknown_place_learns_nothing():
    beliefs = beliefs_after(failed("move", {"location": "shop"}, "road_closed"))
    assert beliefs.holds(road("home", "shop")) is None


def test_look_reports_items_stations_and_roads():
    beliefs = beliefs_after(
        ok(
            "look",
            location="shop",
            items=["bulb"],
            stations=[{"name": "repair_1", "online": False}],
            roads=[{"to": "home", "open": True}, {"to": "park", "open": False}],
        )
    )
    assert beliefs.location == "shop"
    assert beliefs.holds(item_at("bulb", "shop")) is True
    assert beliefs.holds(item_at("lamp", "shop")) is False  # a look lists everything there
    assert beliefs.holds(item_at("lamp", "home")) is None
    assert beliefs.holds(online("repair_1")) is False
    assert beliefs.holds(road("home", "shop")) is True
    assert beliefs.holds(road("park", "shop")) is False


def test_look_uses_the_known_location_when_the_payload_has_none():
    beliefs = beliefs_after(ok("status", location="shop"), ok("look", items=[]))
    assert beliefs.holds(item_at("bulb", "shop")) is False


def test_read_map_reports_items_stations_and_roads():
    beliefs = beliefs_after(
        ok(
            "read_map",
            roads=[{"from": "home", "to": "shop", "open": True}],
            items=[{"item": "bulb", "location": "shop"}],
            stations=[{"name": "repair_1", "location": "garage", "online": True}],
        )
    )
    assert beliefs.holds(road("shop", "home")) is True
    assert beliefs.holds(item_at("bulb", "shop")) is True
    assert beliefs.holds(item_at("bulb", "home")) is None  # the map does not claim absences
    assert beliefs.holds(online("repair_1")) is True


def test_latest_report_wins():
    stale_map = ok("read_map", items=[{"item": "bulb", "location": "shop"}])
    beliefs = beliefs_after(stale_map, ok("look", location="shop", items=[]))
    assert beliefs.holds(item_at("bulb", "shop")) is False

    beliefs.update(stale_map)
    assert beliefs.holds(item_at("bulb", "shop")) is True


def test_pick_and_drop():
    beliefs = beliefs_after(ok("status", location="shop", inventory=[]))
    assert beliefs.holds(carrying("bulb")) is False

    beliefs.update(ok("pick", {"item": "bulb"}))
    assert beliefs.holds(carrying("bulb")) is True
    assert beliefs.holds(item_at("bulb", "shop")) is False

    beliefs.update(ok("move", {"location": "office"}))
    beliefs.update(ok("drop", {"item": "bulb"}))
    assert beliefs.holds(carrying("bulb")) is False
    assert beliefs.holds(item_at("bulb", "office")) is True


def test_failed_pick_and_drop():
    beliefs = beliefs_after(
        ok("status", location="shop"),
        failed("pick", {"item": "bulb"}, "item_not_here"),
        failed("drop", {"item": "lamp"}, "not_carrying"),
    )
    assert beliefs.holds(item_at("bulb", "shop")) is False
    assert beliefs.holds(carrying("bulb")) is None
    assert beliefs.holds(carrying("lamp")) is False


def test_inventory_listing_is_the_servers_word():
    beliefs = beliefs_after(ok("pick", {"item": "bulb"}, inventory=["lamp"]))
    assert beliefs.holds(carrying("bulb")) is False  # same result, listing wins the tie
    assert beliefs.holds(carrying("lamp")) is True


def test_use():
    beliefs = beliefs_after(ok("use", {"station": "repair_1", "items": ["lamp", "bulb"]}, output="fixed_lamp"))
    assert beliefs.holds(online("repair_1")) is True
    assert beliefs.holds(carrying("lamp")) is False and beliefs.holds(carrying("bulb")) is False
    assert beliefs.holds(carrying("fixed_lamp")) is True

    beliefs.update(failed("use", {"station": "repair_1", "items": "lamp"}, "station_offline"))
    assert beliefs.holds(online("repair_1")) is False


def test_other_failures_learn_nothing_new():
    beliefs = beliefs_after(failed("use", {"station": "repair_1", "items": "lamp"}, "missing_items"))
    assert beliefs.holds(online("repair_1")) is None and beliefs.holds(carrying("lamp")) is None


def test_failure_trigger():
    assert failure(ok("move", {"location": "shop"}), step_index=2, call_index=5) is None
    trigger = failure(failed("move", {"location": "shop"}, "road_closed"), step_index=2, call_index=5)
    assert trigger == Trigger("tool_failed", step_index=2, call_index=5, detail="road_closed")


def test_failure_without_a_code():
    trigger = failure(ToolResult("look", {}, ok=False), step_index=0, call_index=0)
    assert trigger.detail == "unknown_error"


def test_contradiction_trigger_names_the_result_that_revealed_it():
    beliefs = beliefs_after(
        ok("status", location="home"),  # call 0
        ok("read_map", roads=[{"from": "home", "to": "shop", "open": True}]),  # call 1
        ok("look", location="home", roads=[{"to": "shop", "open": False}]),  # call 2
    )
    step = PlanStep("move", {"location": "shop"}, [at("home"), road("home", "shop")])
    assert contradiction(step, beliefs, step_index=4) == Trigger(
        "precondition_contradicted", step_index=4, call_index=2, detail="road_open(home, shop)"
    )


def test_unknown_or_satisfied_preconditions_do_not_fire():
    beliefs = beliefs_after(ok("status", location="home"))
    step = PlanStep("move", {"location": "shop"}, [at("home"), road("home", "shop"), online("repair_1")])
    assert contradiction(step, beliefs, step_index=0) is None
    assert contradiction(PlanStep("look"), beliefs, step_index=0) is None


def test_first_contradiction_is_reported():
    beliefs = beliefs_after(ok("status", location="home", inventory=[]))
    step = PlanStep("drop", {"item": "bulb"}, [at("office"), carrying("bulb")])
    assert contradiction(step, beliefs, step_index=0).detail == "at(office)"
