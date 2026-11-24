"""Disruption injector: fires scheduled disruptions at progress-based trigger points.

When the subgoal count reaches a scheduled ``trigger_subgoal`` (right after the write action that
reached it), the injector picks a concrete target and changes the world, so the agent meets the
change on its next action. It never fires once every job is done.

Target rule (FP2)
-----------------
Among candidate resources of the scheduled type, a target is **valid** if applying the
disruption (1) *raises the optimal remaining cost* (so it breaks every optimal plan, not just the
agent's), (2) leaves the task *solvable*, and (3) *affects at least one pending job* (see below),
i.e. it hits a resource some unfinished job requires. All are checked with the exact solver from
the true current state; a candidate whose re-solve is not proven optimal is skipped. Condition (3)
matters: a disruption can raise the joint optimum (by spoiling a shared route) without raising
any single job's own optimum, which would leave its recovery outcome undefined.

Candidates are tried in a seeded random order (seed = scenario seed + disruption index) and the
first valid one is used. That is equivalent to a uniform random choice among all valid targets,
but usually needs only a few re-solves.

If no candidate is valid, the injector **relaxes** to the first candidate (same order) that keeps
the task solvable, marked ``fallback=True``. If none keeps it solvable, the disruption is recorded
as not fired (``reason="no_valid_target"``). Nothing is ever skipped silently.

Candidates per type
-------------------
- road_closure: open roads that are not bridges of the open-road graph (closing a bridge could
  cut the map in two);
- moved_item / stale_map: (item, new location) for items on the ground whose type a pending job
  delivers, consumes, or produces; stale_map leaves the agent's map view unchanged;
- station_offline: online stations of a type some pending craft job needs.

Each fired disruption also records which pending jobs it **affected**: a job is affected if its
own optimal completion cost (solved for that job alone) rises or it becomes unsolvable, i.e. every
optimal way of completing it needed the disrupted resource (used by the recovery-rate metric).
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any

from .maps import bridges_of
from .progress import ProgressTracker
from .recipes import RECIPES
from .seeding import stable_seed
from .solver import SolverConfig, plan, solve
from .tools import dispatch
from .types import AGENT, CONSUMED, DisruptionSpec, DisruptionType, JobKind, Scenario
from .world import World

# Expansion-only budget (no wall-clock limit) so target choice never depends on machine load.
# 300k expansions is roughly 15 s in the worst case; mid-episode re-solves typically need < 1k.
DEFAULT_SOLVER = SolverConfig(max_expansions=300_000, fallback_weight=None)


@dataclass
class DisruptionEvent:
    index: int
    type: DisruptionType
    scheduled_trigger: int
    fired: bool
    step: int | None = None  # episode step after which it fired (or was abandoned)
    fired_at_subgoal: int | None = None
    target: str | None = None  # "road:a|b", "item:x", "station:s"
    detail: dict[str, Any] = field(default_factory=dict)
    cost_before: int | None = None
    cost_after: int | None = None
    fallback: bool = False
    reason: str | None = None  # why it did not fire
    affected_jobs: list[str] = field(default_factory=list)
    candidates_total: int = 0
    candidates_evaluated: int = 0
    candidates_unproven: int = 0
    candidates_unaffecting: int = 0  # raised the joint cost but no single job's
    solve_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["type"] = self.type.value
        d["solve_seconds"] = round(self.solve_seconds, 4)
        return d


# --------------------------------------------------------------------------- candidates


def _needed_types(tracker: ProgressTracker) -> set[str]:
    types: set[str] = set()
    for j in tracker.pending_jobs():
        types.add(j.target_type)
        if j.kind is JobKind.CRAFT:
            types.update(RECIPES[j.target_type].inputs)
    return types


def candidates(world: World, tracker: ProgressTracker, dtype: DisruptionType) -> list[tuple]:
    """Sorted candidate list for a disruption type. Each candidate is a hashable tuple."""
    if dtype is DisruptionType.ROAD_CLOSURE:
        adj = {l: world.open_neighbors(l) for l in sorted(world.location_names)}
        br = set(bridges_of(adj))
        return [("road", a, b) for (a, b) in sorted(world.roads_open)
                if world.roads_open[(a, b)] and (a, b) not in br]
    if dtype in (DisruptionType.MOVED_ITEM, DisruptionType.STALE_MAP):
        needed = _needed_types(tracker)
        out = []
        for item in sorted(world.item_type):
            loc = world.item_location[item]
            if loc in (AGENT, CONSUMED) or world.item_type[item] not in needed:
                continue
            for dest in sorted(world.location_names):
                if dest != loc:
                    out.append(("item", item, dest))
        return out
    if dtype is DisruptionType.STATION_OFFLINE:
        st_types = {
            RECIPES[j.target_type].station_type
            for j in tracker.pending_jobs() if j.kind is JobKind.CRAFT
        }
        return [("station", s) for s in sorted(world.station_type)
                if world.station_online[s] and world.station_type[s] in st_types]
    raise ValueError(dtype)


def apply_candidate(world: World, dtype: DisruptionType, cand: tuple) -> dict[str, Any]:
    """Apply a candidate to ``world`` (mutates it). Returns a JSON-friendly detail dict."""
    kind = cand[0]
    if kind == "road":
        world.close_road(cand[1], cand[2])
        return {"road": [cand[1], cand[2]]}
    if kind == "item":
        item, dest = cand[1], cand[2]
        src = world.item_location[item]
        world.move_item(item, dest, update_map=dtype is not DisruptionType.STALE_MAP)
        return {"item": item, "item_type": world.item_type[item], "from": src, "to": dest,
                "map_updated": dtype is not DisruptionType.STALE_MAP}
    if kind == "station":
        world.set_station_online(cand[1], False)
        return {"station": cand[1], "station_type": world.station_type[cand[1]]}
    raise ValueError(cand)


def target_label(cand: tuple) -> str:
    if cand[0] == "road":
        return f"road:{cand[1]}|{cand[2]}"
    return f"{cand[0]}:{cand[1]}"


# --------------------------------------------------------------------------- injector


class Injector:
    """Holds a scenario's schedule and fires each disruption once its trigger is reached."""

    def __init__(
        self,
        schedule: tuple[DisruptionSpec, ...] | list[DisruptionSpec],
        seed: int,
        solver_config: SolverConfig | None = None,
    ) -> None:
        self.schedule = sorted(schedule, key=lambda d: (d.trigger_subgoal, d.index))
        self.seed = seed
        self.solver_config = solver_config or DEFAULT_SOLVER
        self._next = 0
        self.events: list[DisruptionEvent] = []

    def remaining(self) -> list[DisruptionSpec]:
        return self.schedule[self._next:]

    def fire_due(self, world: World, tracker: ProgressTracker, step: int) -> list[DisruptionEvent]:
        """Fire every scheduled disruption whose trigger has been reached (in order).

        Call after each successful write action. Several may fire in one call if a single action
        crossed several trigger points (e.g. a delivery that also credits inferred subgoals)."""
        fired: list[DisruptionEvent] = []
        while self._next < len(self.schedule):
            spec = self.schedule[self._next]
            if tracker.subgoals_completed() < spec.trigger_subgoal:
                break
            self._next += 1
            if tracker.all_done():
                ev = DisruptionEvent(spec.index, spec.type, spec.trigger_subgoal, False,
                                     step=step, reason="task_complete")
            else:
                ev = self._fire(spec, world, tracker, step)
            self.events.append(ev)
            fired.append(ev)
        return fired

    def finalize(self, step: int | None = None) -> list[DisruptionEvent]:
        """At episode end, record every disruption whose trigger was never reached."""
        out = []
        for spec in self.schedule[self._next:]:
            ev = DisruptionEvent(spec.index, spec.type, spec.trigger_subgoal, False, step=step,
                                 reason="not_reached")
            self.events.append(ev)
            out.append(ev)
        self._next = len(self.schedule)
        return out

    # ------------------------------------------------------------------ internals

    def _fire(self, spec: DisruptionSpec, world: World, tracker: ProgressTracker,
              step: int) -> DisruptionEvent:
        t0 = time.perf_counter()
        ev = DisruptionEvent(spec.index, spec.type, spec.trigger_subgoal, False, step=step,
                             fired_at_subgoal=tracker.subgoals_completed())
        base = solve(world, tracker, self.solver_config)
        ev.cost_before = base.cost
        cands = candidates(world, tracker, spec.type)
        ev.candidates_total = len(cands)
        if base.cost is None or not base.proven_optimal:
            ev.reason = "base_unsolved"
            ev.solve_seconds = time.perf_counter() - t0
            return ev

        rng = random.Random(stable_seed(self.seed, spec.index, "disruption"))
        order = list(cands)
        rng.shuffle(order)
        chosen, chosen_cost, chosen_affected, relaxed = None, None, [], None
        for cand in order:
            w2 = world.clone()
            apply_candidate(w2, spec.type, cand)
            res = solve(w2, tracker, self.solver_config)
            ev.candidates_evaluated += 1
            if res.solvable is False:
                continue
            if relaxed is None and res.solvable:
                relaxed = (cand, res.cost)
            if not res.proven_optimal:
                ev.candidates_unproven += 1
                continue
            if res.cost > base.cost:
                affected = self._affected_jobs(world, w2, tracker)
                if affected:
                    chosen, chosen_cost, chosen_affected = cand, res.cost, affected
                    break
                ev.candidates_unaffecting += 1
        if chosen is None and relaxed is not None:
            chosen, chosen_cost = relaxed
            ev.fallback = True
        if chosen is None:
            ev.reason = "no_valid_target"
            ev.solve_seconds = time.perf_counter() - t0
            return ev

        before = world.clone()
        ev.detail = apply_candidate(world, spec.type, chosen)
        ev.fired = True
        ev.target = target_label(chosen)
        ev.cost_after = chosen_cost
        ev.affected_jobs = (
            chosen_affected if not ev.fallback else self._affected_jobs(before, world, tracker)
        )
        ev.solve_seconds = time.perf_counter() - t0
        return ev

    def _affected_jobs(self, before: World, after: World, tracker: ProgressTracker) -> list[str]:
        affected = []
        for j in tracker.pending_jobs():
            only = tracker.restricted_to({j.id})
            b = solve(before, only, self.solver_config)
            a = solve(after, only, self.solver_config)
            if a.solvable is False or (a.cost is not None and b.cost is not None and a.cost > b.cost):
                affected.append(j.id)
        return affected


# --------------------------------------------------------------------------- schedule replay


def replay_optimal(
    scenario: Scenario,
    schedule: tuple[DisruptionSpec, ...] | None = None,
    solver_config: SolverConfig | None = None,
    max_steps: int = 1000,
) -> tuple[list[DisruptionEvent], int]:
    """Follow an optimal plan from the start, firing disruptions as they come due and re-planning
    optimally after each one (the oracle's behaviour). Returns the injector's events and the
    number of write actions taken.

    Used by the generator to check that every scheduled disruption fires and breaks the optimal
    plan (FP2 section 4.4), and by validation (step 11)."""
    w = World.from_scenario(scenario)
    t = ProgressTracker(scenario.jobs)
    inj = Injector(scenario.schedule if schedule is None else schedule, scenario.seed,
                   solver_config)
    calls = plan(w, t) or []
    steps = 0
    while not t.all_done():
        if not calls:
            break  # unsolvable or unproven: caller sees missing completion
        name, args = calls.pop(0)
        r = dispatch(w, name, args)
        steps += 1
        if not r.ok:
            raise AssertionError(f"{scenario.id}: optimal plan step failed: {r.text}")
        t.observe(r.tool, r.ok, r.data)
        if inj.fire_due(w, t, steps):
            calls = plan(w, t) or []
        if steps >= max_steps:
            break
    inj.finalize(steps)
    return inj.events, steps


def all_fired_strictly(events: list[DisruptionEvent]) -> bool:
    return all(e.fired and not e.fallback for e in events)
