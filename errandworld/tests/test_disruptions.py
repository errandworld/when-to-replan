"""Disruption injector: candidates, effects, the target rule, fallbacks, and dataset replay."""

import glob
import json
from pathlib import Path

import pytest

from conftest import make_tiny_scenario
from errandworld.disruptions import (
    Injector,
    all_fired_strictly,
    apply_candidate,
    candidates,
    replay_optimal,
)
from errandworld.progress import ProgressTracker
from errandworld.scenario_io import load_scenario
from errandworld.solver import optimal_cost
from errandworld.tools import dispatch
from errandworld.types import (
    DisruptionSpec,
    DisruptionType as DT,
    Item,
    Job,
    JobKind,
    Location,
    MapSpec,
    Road,
    Scenario,
)
from errandworld.world import World

ROOT = Path(__file__).resolve().parents[1]


def fresh(s):
    return World.from_scenario(s), ProgressTracker(s.jobs)


def act(w, t, name, **args):
    r = dispatch(w, name, args)
    assert r.ok, r.text
    t.observe(r.tool, r.ok, r.data)
    return r


# ---------------------------------------------------------------- candidates


def test_road_candidates_exclude_bridges(tiny_scenario):
    w, t = fresh(tiny_scenario)
    roads = candidates(w, t, DT.ROAD_CLOSURE)
    assert ("road", "market", "workshop") not in roads  # bridge: workshop is a leaf
    assert ("road", "home", "market") in roads
    w.close_road("home", "park")  # now the remaining cycle is broken: everything is a bridge
    assert candidates(w, t, DT.ROAD_CLOSURE) == []


def test_item_candidates_are_needed_items_on_the_ground(tiny_scenario):
    w, t = fresh(tiny_scenario)
    items = {c[1] for c in candidates(w, t, DT.MOVED_ITEM)}
    assert items == {"battery_1", "lamp_base_1", "bulb_1"}  # ink_1 is a distractor
    act(w, t, "move", location="market")
    act(w, t, "pick", item="battery_1")
    items = {c[1] for c in candidates(w, t, DT.STALE_MAP)}
    assert "battery_1" not in items  # carried items cannot be moved
    # every other location is a destination
    dests = {c[2] for c in candidates(w, t, DT.MOVED_ITEM) if c[1] == "bulb_1"}
    assert dests == {"home", "market", "office", "park"}


def test_station_candidates_only_for_pending_craft_types(tiny_scenario):
    w, t = fresh(tiny_scenario)
    assert candidates(w, t, DT.STATION_OFFLINE) == [
        ("station", "repair_bench_1"), ("station", "repair_bench_2"),
    ]
    only_deliver = make_tiny_scenario(jobs=(Job("job_1", JobKind.DELIVER, "battery", "office"),),
                                      total_subgoals=2, schedule=(), disruption_level=0)
    w2, t2 = fresh(only_deliver)
    assert candidates(w2, t2, DT.STATION_OFFLINE) == []


def test_apply_candidate_moved_vs_stale(tiny_scenario):
    w, _ = fresh(tiny_scenario)
    d = apply_candidate(w, DT.MOVED_ITEM, ("item", "bulb_1", "park"))
    assert d == {"item": "bulb_1", "item_type": "bulb", "from": "workshop", "to": "park",
                 "map_updated": True}
    assert w.map_item_location("bulb_1") == "park"
    w2, _ = fresh(tiny_scenario)
    apply_candidate(w2, DT.STALE_MAP, ("item", "bulb_1", "park"))
    assert w2.item_location["bulb_1"] == "park" and w2.map_item_location("bulb_1") == "workshop"


# ---------------------------------------------------------------- firing


def scenario_with(schedule, **kw):
    return make_tiny_scenario(schedule=tuple(schedule), disruption_level=len(schedule), **kw)


def ring_scenario(schedule, seed=5):
    """Ring r0..r5. Start r0 with battery, lamp_base, bulb. Deliver battery to r1; craft a lamp
    at a repair_bench (r1 near, r3 far) and deliver it to r1. Every disruption type has targets
    that clearly raise the optimal cost."""
    from errandworld.types import Station

    locs = [f"r{i}" for i in range(6)]
    m = MapSpec("ring", 0, tuple(Location(l, l) for l in locs),
                tuple(Road(locs[i], locs[(i + 1) % 6]) for i in range(6)),
                (Station("repair_bench_1", "repair_bench", "r1"),
                 Station("repair_bench_2", "repair_bench", "r3")))
    return Scenario(
        id="ring", seed=seed, set_name="unit", map=m, start_location="r0",
        items=(Item("battery_1", "battery", "r0"), Item("lamp_base_1", "lamp_base", "r0"),
               Item("bulb_1", "bulb", "r0")),
        jobs=(Job("job_1", JobKind.DELIVER, "battery", "r1"),
              Job("job_2", JobKind.CRAFT, "lamp", "r1")),
        schedule=tuple(schedule), capacity=3, total_subgoals=6, disruption_level=len(schedule),
    )


@pytest.mark.parametrize("dtype", list(DT))
def test_each_type_fires_strictly_and_raises_cost(dtype):
    s = ring_scenario([DisruptionSpec(0, dtype, 1)])
    w, t = fresh(s)
    inj = Injector(s.schedule, s.seed)
    assert optimal_cost(w, t) == 7  # 3 picks, move, use, 2 drops
    assert inj.fire_due(w, t, step=0) == []  # nothing completed yet
    act(w, t, "pick", item="battery_1")
    (ev,) = inj.fire_due(w, t, step=1)
    assert ev.fired and not ev.fallback and ev.reason is None
    assert ev.fired_at_subgoal == 1 and ev.step == 1
    assert ev.cost_after > ev.cost_before == 6
    assert optimal_cost(w, t) == ev.cost_after  # the world really changed
    assert ev.affected_jobs  # at least one job's own optimum rose
    assert ev.candidates_evaluated >= 1 and ev.target is not None
    assert inj.fire_due(w, t, step=3) == []  # fires only once


def test_target_choice_is_deterministic():
    s = ring_scenario([DisruptionSpec(0, DT.MOVED_ITEM, 1)])
    targets = set()
    for _ in range(3):
        w, t = fresh(s)
        inj = Injector(s.schedule, s.seed)
        act(w, t, "pick", item="battery_1")
        ev = inj.fire_due(w, t, 1)[0]
        targets.add((ev.target, ev.detail["to"]))
    assert len(targets) == 1


def test_seed_changes_choice_distribution():
    seen = set()
    for seed in range(20):
        s = ring_scenario([DisruptionSpec(0, DT.MOVED_ITEM, 1)], seed=seed)
        w, t = fresh(s)
        inj = Injector(s.schedule, seed)
        act(w, t, "pick", item="battery_1")
        ev = inj.fire_due(w, t, 1)[0]
        assert ev.fired and not ev.fallback
        seen.add((ev.target, ev.detail["to"]))
    assert len(seen) > 3  # uniform choice among many valid (item, destination) pairs


def test_several_due_disruptions_fire_in_order():
    s = scenario_with([DisruptionSpec(0, DT.ROAD_CLOSURE, 1), DisruptionSpec(1, DT.MOVED_ITEM, 2)])
    w, t = fresh(s)
    inj = Injector(s.schedule, s.seed)
    act(w, t, "move", location="market")
    act(w, t, "pick", item="battery_1")
    act(w, t, "pick", item="lamp_base_1")  # count is now 2: both are due
    evs = inj.fire_due(w, t, step=3)
    assert [e.index for e in evs] == [0, 1] and all(e.fired for e in evs)


def test_never_fires_after_task_complete_and_finalize():
    s = scenario_with([DisruptionSpec(0, DT.ROAD_CLOSURE, 5), DisruptionSpec(1, DT.MOVED_ITEM, 5)])
    w, t = fresh(s)
    inj = Injector(s.schedule, s.seed)
    for name, args in [("move", {"location": "market"}), ("pick", {"item": "battery_1"})]:
        act(w, t, name, **args)
    assert inj.fire_due(w, t, 2) == []
    leftover = inj.finalize(step=2)
    assert [e.reason for e in leftover] == ["not_reached", "not_reached"]
    assert len(inj.events) == 2 and inj.remaining() == []


def test_task_complete_reason():
    s = scenario_with([DisruptionSpec(0, DT.ROAD_CLOSURE, 1)])
    w, t = fresh(s)
    inj = Injector(s.schedule, s.seed)
    # force "all done" before the injector looks
    for j in t.jobs:
        t._done[j.id] = True
    t._count = 6
    (ev,) = inj.fire_due(w, t, 1)
    assert not ev.fired and ev.reason == "task_complete"


def square_scenario(schedule):
    """a-b, a-c, b-d, c-d: two equally short routes from a to d."""
    m = MapSpec("sq", 0, tuple(Location(x, x) for x in "abcd"),
                (Road("a", "b"), Road("a", "c"), Road("b", "d"), Road("c", "d")), ())
    return Scenario(
        id="sq", seed=1, set_name="unit", map=m, start_location="a",
        items=(Item("battery_1", "battery", "a"),),
        jobs=(Job("job_1", JobKind.DELIVER, "battery", "d"),),
        schedule=tuple(schedule), capacity=3, total_subgoals=2, disruption_level=len(schedule),
    )


def test_fallback_when_no_target_raises_cost():
    s = square_scenario([DisruptionSpec(0, DT.ROAD_CLOSURE, 1)])
    w, t = fresh(s)
    inj = Injector(s.schedule, s.seed)
    act(w, t, "pick", item="battery_1")
    (ev,) = inj.fire_due(w, t, 1)
    assert ev.fired and ev.fallback  # any single closure leaves an equally short route
    assert ev.cost_after == ev.cost_before == 3
    assert ev.candidates_evaluated == ev.candidates_total == 4


def test_not_fired_when_no_candidate():
    s = square_scenario([DisruptionSpec(0, DT.STATION_OFFLINE, 1)])  # no craft jobs, no stations
    w, t = fresh(s)
    inj = Injector(s.schedule, s.seed)
    act(w, t, "pick", item="battery_1")
    (ev,) = inj.fire_due(w, t, 1)
    assert not ev.fired and ev.reason == "no_valid_target" and ev.candidates_total == 0
    assert all_fired_strictly([ev]) is False


def test_event_to_dict_is_json():
    s = scenario_with([DisruptionSpec(0, DT.STALE_MAP, 1)])
    w, t = fresh(s)
    inj = Injector(s.schedule, s.seed)
    act(w, t, "move", location="market")
    act(w, t, "pick", item="battery_1")
    d = inj.fire_due(w, t, 2)[0].to_dict()
    json.dumps(d)
    assert d["type"] == "stale_map" and d["detail"]["map_updated"] is False


# ---------------------------------------------------------------- replay & datasets


def test_replay_optimal_on_tiny():
    s = scenario_with([DisruptionSpec(0, DT.ROAD_CLOSURE, 2)])
    events, steps = replay_optimal(s)
    assert all_fired_strictly(events) and steps > 12  # the closure forced a detour


def dataset_files(set_name):
    return sorted(glob.glob(str(ROOT / "data/scenarios" / set_name / f"{set_name}_L*_D*_*.json")))


@pytest.mark.parametrize("set_name", ["dev", "pilot", "test"])
def test_committed_reference_replays_are_strict(set_name):
    files = dataset_files(set_name)
    if not files:
        pytest.skip("dataset not generated")
    for f in files:
        s = load_scenario(f)
        rep = s.metadata["reference_replay"]
        assert len(rep) == len(s.schedule), s.id
        for e in rep:
            assert e["cost_after"] > e["cost_before"] and e["affected_jobs"], (s.id, e)
        assert len({d.type for d in s.schedule}) == len(s.schedule), s.id  # distinct types


@pytest.mark.parametrize("path", dataset_files("pilot")[::5])
def test_pilot_replay_reproduces(path):
    """Re-running the replay reproduces the recorded targets (determinism end to end)."""
    s = load_scenario(path)
    events, _ = replay_optimal(s)
    assert all_fired_strictly(events)
    assert [e.target for e in events] == [e["target"] for e in s.metadata["reference_replay"]]
