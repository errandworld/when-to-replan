"""Solver correctness.

The key test compares A* (type abstraction + pruning + heuristic) against an exhaustive
breadth-first search over the *concrete* World (real item ids, real tool calls, no abstraction,
no heuristic) on hundreds of random micro-worlds. Equal costs everywhere means the abstraction is
faithful, the prunings keep an optimal plan, and the heuristic never overestimates.
"""

import os
import random
from collections import deque

import pytest

from conftest import make_tiny_map, make_tiny_scenario
from errandworld.progress import ProgressTracker
from errandworld.recipes import BASE_ITEM_TYPES, RECIPES, STATION_TYPES, crafted_input, tier
from errandworld.scenario_io import scenario_problems
from errandworld.solver import SolverConfig, concretize, is_solvable, optimal_cost, plan, solve
from errandworld.tools import dispatch
from errandworld.types import Item, Job, JobKind, Location, MapSpec, Road, Scenario, Station
from errandworld.world import World


def fresh(scenario):
    return World.from_scenario(scenario), ProgressTracker(scenario.jobs)


# ---------------------------------------------------------------- brute force reference


def concrete_actions(w: World):
    here = w.agent_location
    for n in w.open_neighbors(here):
        yield "move", {"location": n}
    if len(w.inventory) < w.capacity:
        for i in w.items_at(here):
            yield "pick", {"item": i}
    for i in sorted(set(w.inventory)):
        yield "drop", {"item": i}
    inv = sorted(w.inventory)
    for s in w.stations_at(here):
        for a in range(len(inv)):
            for b in range(a + 1, len(inv)):
                yield "use", {"station": s, "items": [inv[a], inv[b]]}


def brute_force_cost(world: World, tracker: ProgressTracker, max_states=400_000):
    """Exhaustive BFS over concrete states. Returns cost, None if unsolvable."""

    def key(w, t):
        snap = w.snapshot()
        # Item ids of crafted outputs depend on history; key on types to merge equivalent states.
        items = sorted((it["type"], it["location"]) for it in snap["items"])
        inv = sorted(w.item_type[i] for i in w.inventory)
        return (snap["agent_location"], tuple(inv), tuple(items), t.state_key()[1])

    if tracker.all_done():
        return 0
    seen = {key(world, tracker)}
    q = deque([(world, tracker, 0)])
    while q:
        w, t, g = q.popleft()
        for name, args in concrete_actions(w):
            w2, t2 = w.clone(), t.clone()
            r = dispatch(w2, name, args)
            if not r.ok:  # e.g. offline station, inputs matching no recipe
                continue
            t2.observe(r.tool, r.ok, r.data)
            if t2.all_done():
                return g + 1
            k = key(w2, t2)
            if k in seen:
                continue
            seen.add(k)
            if len(seen) > max_states:
                pytest.skip("micro-world too large for brute force")
            q.append((w2, t2, g + 1))
    return None


# ---------------------------------------------------------------- random micro-worlds


def random_micro_scenario(rng: random.Random) -> Scenario:
    while True:
        n = rng.randint(3, 6)
        locs = [f"l{i}" for i in range(n)]
        roads = set()
        for i in range(1, n):  # random spanning tree
            roads.add(tuple(sorted((locs[i], locs[rng.randrange(i)]))))
        for _ in range(rng.randint(0, 2)):
            a, b = rng.sample(locs, 2)
            roads.add(tuple(sorted((a, b))))
        road_objs = tuple(Road(a, b, rng.random() > 0.1) for a, b in sorted(roads))
        st_type = rng.choice(STATION_TYPES)
        stations = tuple(
            Station(f"{st_type}_{k}", st_type, rng.choice(locs), rng.random() > 0.15)
            for k in range(1, rng.randint(1, 2) + 1)
        )
        recipes = sorted(o for o, r in RECIPES.items() if r.station_type == st_type)
        jobs, items = [], []
        counter = [0]

        def add_item(t):
            counter[0] += 1
            items.append(Item(f"{t}_{counter[0]}", t, rng.choice(locs)))

        n_jobs = rng.randint(1, 3)
        for k in range(n_jobs):
            jid = f"job_{k + 1}"
            roll = rng.random()
            if roll < 0.45:
                t = rng.choice(BASE_ITEM_TYPES[:4])  # small pool -> shared types happen
                jobs.append(Job(jid, JobKind.DELIVER, t, rng.choice(locs)))
                if rng.random() < 0.7:
                    add_item(t)
            elif roll < 0.85 or k == 0:
                out = rng.choice([o for o in recipes if tier(o) == 1])
                jobs.append(Job(jid, JobKind.CRAFT, out, rng.choice(locs)))
                for t in RECIPES[out].inputs:
                    add_item(t)
            else:
                producers = [j for j in jobs if j.kind is JobKind.CRAFT and tier(j.target_type) == 1]
                t2 = [o for o in recipes if tier(o) == 2 and crafted_input(o) in
                      {p.target_type for p in producers}]
                if not t2:
                    continue
                out = t2[0]
                prod = next(p for p in producers if p.target_type == crafted_input(out))
                jobs.append(Job(jid, JobKind.CRAFT, out, rng.choice(locs), (prod.id,)))
                for t in RECIPES[out].inputs:
                    if t not in RECIPES:
                        add_item(t)
        for _ in range(rng.randint(0, 2)):  # extras and distractors
            add_item(rng.choice(BASE_ITEM_TYPES[:6]))
        s = Scenario(
            id="micro", seed=0, set_name="unit",
            map=MapSpec("m", 0, tuple(Location(l, l) for l in locs), road_objs, stations),
            start_location=rng.choice(locs), items=tuple(items), jobs=tuple(jobs), schedule=(),
            capacity=rng.choice([2, 3]),
        )
        if not scenario_problems(s):
            return s


# Default: 80 seeds (~30 s). Full sweep: ERRANDWORLD_FULL_SOLVER_TESTS=1 (500 seeds).
N_SEEDS = 500 if os.environ.get("ERRANDWORLD_FULL_SOLVER_TESTS") else 80


@pytest.mark.parametrize("seed", range(N_SEEDS))
def test_astar_matches_brute_force(seed):
    s = random_micro_scenario(random.Random(seed))
    w, t = fresh(s)
    res = solve(w, t, SolverConfig(fallback_weight=None))
    expected = brute_force_cost(w, t)
    assert res.proven_optimal
    assert res.cost == expected, (seed, res.cost, expected)
    assert res.solvable == (expected is not None)
    if expected is not None:
        # Executing the concrete plan completes every job in exactly `cost` write actions.
        calls = concretize(w, res.actions)
        assert len(calls) == expected
        for name, args in calls:
            r = dispatch(w, name, args)
            assert r.ok
            t.observe(r.tool, r.ok, r.data)
        assert t.all_done()


# ---------------------------------------------------------------- hand-checked cases


def test_tiny_scenario_optimal_cost(tiny_scenario):
    # home->market, pick battery, pick lamp_base, ->workshop, pick bulb, use, ->market->office,
    # drop battery, ->market->home, drop lamp  = 6 moves + 3 picks + 1 use + 2 drops = 12
    w, t = fresh(tiny_scenario)
    res = solve(w, t)
    assert res.cost == 12 and res.proven_optimal and res.solvable
    assert res.cost == brute_force_cost(w, t)


def test_plan_executes_to_success(tiny_scenario):
    w, t = fresh(tiny_scenario)
    calls = plan(w, t)
    for name, args in calls:
        r = dispatch(w, name, args)
        assert r.ok, r.text
        t.observe(r.tool, r.ok, r.data)
    assert t.all_done() and len(calls) == 12


def test_one_item_can_serve_two_deliveries_via_latching():
    """Latched completion: deliver, re-pick, deliver again. Cost must not assume 2 items."""
    s = make_tiny_scenario(
        items=(Item("battery_1", "battery", "market"),),
        jobs=(
            Job("job_a", JobKind.DELIVER, "battery", "office"),
            Job("job_b", JobKind.DELIVER, "battery", "home"),
        ),
        schedule=(), disruption_level=0, total_subgoals=4,
    )
    w, t = fresh(s)
    # home->market pick ->office drop pick ->market->home drop : 4 moves + 2 picks + 2 drops
    assert optimal_cost(w, t) == 8 == brute_force_cost(w, t)


def test_cost_decreases_along_an_optimal_plan(tiny_scenario):
    w, t = fresh(tiny_scenario)
    calls = plan(w, t)
    for k, (name, args) in enumerate(calls):
        assert optimal_cost(w, t) == len(calls) - k
        r = dispatch(w, name, args)
        t.observe(r.tool, r.ok, r.data)
    assert optimal_cost(w, t) == 0


def test_road_closure_raises_cost(tiny_scenario):
    w, t = fresh(tiny_scenario)
    base = optimal_cost(w, t)
    w.close_road("market", "workshop")  # workshop is a leaf: the bulb becomes unreachable
    assert is_solvable(w, t) is False
    w2, t2 = fresh(tiny_scenario)
    w2.close_road("home", "market")
    assert optimal_cost(w2, t2) > base


def test_offline_station_uses_alternative(tiny_scenario):
    w, t = fresh(tiny_scenario)
    w.set_station_online("repair_bench_1", False)  # must craft at park instead
    res = solve(w, t)
    # Crafting at park on the way home (office -> park -> home) is still 12 actions.
    assert res.solvable and res.cost == 12 == brute_force_cost(w, t)
    assert ("move", "park") in res.actions
    w.set_station_online("repair_bench_2", False)  # no bench left at all
    assert is_solvable(w, t) is False


def test_unsolvable_when_item_missing(tiny_scenario):
    w, t = fresh(tiny_scenario)
    w.item_location["bulb_1"] = "consumed"  # simulate a lost input
    res = solve(w, t)
    assert res.solvable is False and res.cost is None and res.proven_optimal


def test_solver_does_not_mutate_world(tiny_scenario):
    w, t = fresh(tiny_scenario)
    before, tk = w.snapshot(), t.state_key()
    solve(w, t)
    plan(w, t)
    assert w.snapshot() == before and t.state_key() == tk


def test_budget_exhaustion_reports_unknown(tiny_scenario):
    w, t = fresh(tiny_scenario)
    res = solve(w, t, SolverConfig(max_expansions=3, fallback_weight=None))
    assert res.cost is None and res.solvable is None and not res.proven_optimal
    assert res.stats.budget_hit


def test_fallback_marks_result_not_proven(tiny_scenario):
    w, t = fresh(tiny_scenario)
    res = solve(w, t, SolverConfig(max_expansions=15, fallback_weight=50.0))
    if res.cost is not None:  # greedy search may succeed within the tiny budget
        assert not res.proven_optimal and res.stats.attempts == 2


def test_solve_is_deterministic(tiny_scenario):
    w, t = fresh(tiny_scenario)
    a, b = solve(w, t), solve(w, t)
    assert a.actions == b.actions and a.stats.expanded == b.stats.expanded


def test_mid_episode_with_done_jobs(tiny_scenario):
    w, t = fresh(tiny_scenario)
    for name, args in [("move", {"location": "market"}), ("pick", {"item": "battery_1"}),
                       ("move", {"location": "office"}), ("drop", {"item": "battery_1"})]:
        r = dispatch(w, name, args)
        t.observe(r.tool, r.ok, r.data)
    assert t.job_done("job_1")
    assert optimal_cost(w, t) == brute_force_cost(w, t)


@pytest.mark.parametrize("seed", range(40))
def test_budget_bracket_contains_true_optimum(seed):
    """With a tiny budget, lower_bound <= optimum <= upper_bound must still hold."""
    s = random_micro_scenario(random.Random(1000 + seed))
    w, t = fresh(s)
    exact = solve(w, t, SolverConfig(fallback_weight=None))
    if not exact.solvable:
        return
    res = solve(w, t, SolverConfig(max_expansions=5, fallback_weight=3.0))
    if res.lower_bound is not None:
        assert res.lower_bound <= exact.cost
    if res.upper_bound is not None:
        assert exact.cost <= res.upper_bound == res.cost
        assert res.proven_optimal == (res.lower_bound == res.upper_bound)
