"""Scenario generator.

``generate_scenario`` samples jobs and item placements on a fixed map, solves the result
exactly, and keeps it only if its proven optimal length falls within ±10% of the target length
level. It resamples (with a new attempt seed) until a scenario is accepted, recording every
rejection by reason.

Design (ARCHITECTURE.md, D6)
----------------------------
- Length levels 10/20/30 use 2/3/4 jobs; the rest of the length comes from travel, controlled
  by how far from a random "hub" items and destinations are placed (``radii``).
- Job **composition is fixed, not sampled**: ``planned_composition`` gives each scenario an
  exact number of craft and dependent jobs (half the jobs are craft jobs; the 3-job level
  alternates 1 and 2 by scenario index). Sampling it instead lets the length filter select it:
  craft jobs are longer, so a calibration run accepted 27% craft jobs at length 10 vs 60% at
  length 30, confounding length with task structure. With composition fixed, the filter can only
  select item/destination placement. Dependent (tier-2) jobs are off by default because a 2-job
  task cannot hold one without becoming all-craft; set ``dependent_jobs`` to enable them.
- Rules that keep tasks well-posed:
    * deliver targets are distinct, and never a craft input in the same scenario (otherwise one
      item could be delivered and then consumed or re-delivered via latched completion);
    * craft outputs are distinct;
    * every needed item exists exactly once (so a moved or stale item really matters);
    * a deliver job's item never starts at its own destination;
    * a few distractor items use types no job needs.
- Only *proven* optima are accepted (exact A* with an offline budget). Unproven samples are
  counted as rejections (``unproven``) so any selection effect is reported, not hidden.
- The disruption schedule's *types* are supplied by the caller (balanced within each cell by
  ``balanced_disruption_types``); trigger points are evenly spaced over the scenario's subgoals.
- **Schedule replay** (FP2 section 4.4): the scenario is accepted only if, along the reference
  trajectory (optimal plan, re-planned optimally after each disruption), every disruption fires
  and raises the optimal remaining cost. If the given order of types fails, other orderings of
  the same types over the same trigger points are tried first (keeps the cell's type balance
  exact); only then is the placement resampled (rejection reason ``schedule_replay``).

Determinism: attempt ``k`` of a scenario uses ``random.Random(seed * 1000 + k)``; nothing else is
random.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import random
from collections import deque
from dataclasses import dataclass, field

from .progress import ProgressTracker, count_subgoals
from .recipes import BASE_ITEM_TYPES, RECIPES, crafted_input, tier
from .scenario_io import check_scenario
from .seeding import stable_seed  # noqa: F401  (re-exported)
from .solver import SolverConfig, solve
from .types import (
    DisruptionSpec,
    DisruptionType,
    Item,
    Job,
    JobKind,
    MapSpec,
    Scenario,
)
from .world import World

GENERATOR_VERSION = 4

LENGTH_LEVELS = (10, 20, 30)
DISRUPTION_LEVELS = (0, 1, 3)


@dataclass(frozen=True)
class GeneratorConfig:
    jobs_per_level: dict[int, int] = field(default_factory=lambda: {10: 2, 20: 3, 30: 4})
    # Candidate placement radii (in roads from the hub) per level; one is drawn per attempt.
    radii: dict[int, tuple[int, ...]] = field(
        default_factory=lambda: {10: (1, 2), 20: (2, 3, 4), 30: (3, 4, 8)}
    )
    craft_share: float = 0.5  # fraction of jobs that are craft jobs (exact per level, see below)
    dependent_jobs: dict[int, int] = field(default_factory=lambda: {10: 0, 20: 0, 30: 0})
    n_distractors: tuple[int, int] = (2, 3)  # inclusive range
    tolerance: float = 0.10
    capacity: int = 3
    max_attempts: int = 2000
    solver: SolverConfig = field(
        default_factory=lambda: SolverConfig(
            # Expansion-only (machine-independent). The largest accepted solve used ~100k.
            max_expansions=2_000_000, fallback_weight=None
        )
    )

    def planned_composition(self, length_level: int, index: int) -> tuple[int, int]:
        """(craft jobs, dependent jobs) for the ``index``-th scenario of a length level.

        When ``jobs * craft_share`` is fractional, scenarios alternate between rounding down and
        up so the cell's average craft share is exact."""
        exact = self.jobs_per_level[length_level] * self.craft_share
        n_craft = math.floor(exact) + (1 if (index % 2 == 1 and exact % 1) else 0)
        return n_craft, self.dependent_jobs.get(length_level, 0)

    def band(self, length_level: int) -> tuple[int, int]:
        lo = math.ceil(length_level * (1 - self.tolerance) - 1e-9)
        hi = math.floor(length_level * (1 + self.tolerance) + 1e-9)
        return lo, hi


class GenerationError(RuntimeError):
    pass


# --------------------------------------------------------------------------- seeds & schedules


def trigger_points(total_subgoals: int, k: int) -> list[int]:
    """``k`` evenly spaced, strictly increasing trigger points in [1, total_subgoals - 1]."""
    if k == 0:
        return []
    if total_subgoals - 1 < k:
        raise GenerationError(f"{total_subgoals} subgoals cannot hold {k} disruptions")
    pts: list[int] = []
    for i in range(1, k + 1):
        p = max(1, round(i * total_subgoals / (k + 1)))
        if pts and p <= pts[-1]:
            p = pts[-1] + 1
        pts.append(p)
    # Shift down if bumping pushed the last point too far.
    over = pts[-1] - (total_subgoals - 1)
    if over > 0:
        pts = [p - over for p in pts]
    assert pts[0] >= 1 and pts[-1] <= total_subgoals - 1 and pts == sorted(set(pts))
    return pts


def balanced_disruption_types(n: int, seed: int) -> list[DisruptionType]:
    """``n`` disruption types in shuffled blocks of all four, so counts differ by at most 1."""
    rng = random.Random(seed)
    types = list(DisruptionType)
    out: list[DisruptionType] = []
    while len(out) < n:
        block = list(types)
        rng.shuffle(block)
        out.extend(block)
    return out[:n]


def cell_disruption_types(n: int, k: int, seed: int) -> list[list[DisruptionType]]:
    """Disruption types for the ``n`` scenarios of a cell with ``k`` disruptions each.

    Each scenario gets ``k`` *distinct* types (duplicates can be infeasible: e.g. two
    station_offline disruptions on a task with one craft job and two stations of its type leave
    no way to craft). Scenarios cycle through all k-subsets of the four types in shuffled blocks,
    so every type appears equally often within the cell (counts differ by at most one), and the
    order of types within each scenario is shuffled.
    """
    if k == 0:
        return [[] for _ in range(n)]
    types = list(DisruptionType)
    if k > len(types):
        raise ValueError(f"cannot give {k} distinct disruption types")
    rng = random.Random(seed)
    subsets = [list(c) for c in itertools.combinations(types, k)]
    out: list[list[DisruptionType]] = []
    while len(out) < n:
        block = [list(s) for s in subsets]
        rng.shuffle(block)
        for s in block:
            rng.shuffle(s)
        out.extend(block)
    return out[:n]


# --------------------------------------------------------------------------- sampling


def _ball(m: MapSpec, center: str, radius: int) -> list[str]:
    nbrs: dict[str, list[str]] = {l.id: [] for l in m.locations}
    for r in m.roads:
        nbrs[r.a].append(r.b)
        nbrs[r.b].append(r.a)
    dist = {center: 0}
    q = deque([center])
    while q:
        u = q.popleft()
        for v in sorted(nbrs[u]):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return sorted(l for l, d in dist.items() if d <= radius)


def _sample_jobs(
    rng: random.Random, n_jobs: int, n_craft: int, n_dependent: int
) -> list[tuple[str, str, int | None]]:
    """(kind, target_type, producer_index_or_None) per job, in job order.

    Exactly ``n_craft`` craft jobs, of which ``n_dependent`` are tier-2 jobs each consuming the
    output of a distinct tier-1 craft job. Deliver targets are chosen last, from base types no
    craft job consumes.
    """
    if not 0 <= n_craft <= n_jobs or not 0 <= 2 * n_dependent <= n_craft:
        raise GenerationError(f"infeasible composition: {n_jobs} jobs, {n_craft} craft, "
                              f"{n_dependent} dependent")
    tier1 = sorted(o for o in RECIPES if tier(o) == 1)
    tier2 = sorted(o for o in RECIPES if tier(o) == 2)
    kinds = ["craft"] * n_craft + ["deliver"] * (n_jobs - n_craft)
    rng.shuffle(kinds)
    craft_idx = [i for i, k in enumerate(kinds) if k == "craft"]

    specs: list[list] = [[k, None, None] for k in kinds]
    # Dependent pairs first: a tier-1 producer and the tier-2 consumer that comes after it.
    feedable = [o for o in tier1 if any(crafted_input(t2) == o for t2 in tier2)]
    rng.shuffle(feedable)
    used_outputs: set[str] = set()
    for d in range(n_dependent):
        producer, consumer = craft_idx[2 * d], craft_idx[2 * d + 1]
        out1 = feedable[d]
        out2 = next(t2 for t2 in tier2 if crafted_input(t2) == out1)
        specs[producer][1] = out1
        specs[consumer][1], specs[consumer][2] = out2, producer
        used_outputs |= {out1, out2}
    free = [o for o in tier1 if o not in used_outputs]
    rng.shuffle(free)
    for i in craft_idx[2 * n_dependent:]:
        if not free:
            raise GenerationError("ran out of distinct craft outputs")
        specs[i][1] = free.pop()

    craft_inputs = {t for k, out, _ in specs if k == "craft" for t in RECIPES[out].inputs}
    pool = [t for t in BASE_ITEM_TYPES if t not in craft_inputs]
    rng.shuffle(pool)
    for s in specs:
        if s[0] == "deliver":
            if not pool:
                raise GenerationError("ran out of deliverable item types")
            s[1] = pool.pop()
    return [tuple(s) for s in specs]  # type: ignore[misc]


def sample_candidate(
    m: MapSpec,
    length_level: int,
    rng: random.Random,
    cfg: GeneratorConfig,
    n_craft: int,
    n_dependent: int,
) -> tuple[str, list[Item], list[Job], int]:
    """One unsolved candidate: (start location, items, jobs, radius)."""
    radius = rng.choice(cfg.radii[length_level])
    hub = rng.choice(sorted(l.id for l in m.locations))
    area = _ball(m, hub, radius)
    start = rng.choice(area)

    specs = _sample_jobs(rng, cfg.jobs_per_level[length_level], n_craft, n_dependent)
    jobs: list[Job] = []
    items: list[Item] = []
    counts: dict[str, int] = {}

    def add_item(t: str, avoid: str | None = None) -> None:
        counts[t] = counts.get(t, 0) + 1
        choices = [l for l in area if l != avoid] or [l for l in sorted(x.id for x in m.locations) if l != avoid]
        items.append(Item(f"{t}_{counts[t]}", t, rng.choice(choices)))

    for i, (kind, target, producer) in enumerate(specs):
        jid = f"job_{i + 1}"
        dest = rng.choice(area)
        if kind == "deliver":
            jobs.append(Job(jid, JobKind.DELIVER, target, dest))
            add_item(target, avoid=dest)
        else:
            deps = (f"job_{producer + 1}",) if producer is not None else ()
            jobs.append(Job(jid, JobKind.CRAFT, target, dest, deps))
            for t in RECIPES[target].inputs:
                if t not in RECIPES:  # crafted inputs come from the producer job
                    add_item(t)

    used_types = {it.type for it in items} | {
        t for j in jobs if j.kind is JobKind.CRAFT for t in RECIPES[j.target_type].inputs
    }
    spare = [t for t in BASE_ITEM_TYPES if t not in used_types and t not in {j.target_type for j in jobs}]
    lo, hi = cfg.n_distractors
    for t in rng.sample(spare, min(len(spare), rng.randint(lo, hi))):
        counts[t] = counts.get(t, 0) + 1
        items.append(Item(f"{t}_{counts[t]}", t, rng.choice(sorted(x.id for x in m.locations))))
    return start, items, jobs, radius


def composition(jobs: list[Job] | tuple[Job, ...]) -> dict[str, int]:
    return {
        "jobs": len(jobs),
        "deliver": sum(j.kind is JobKind.DELIVER for j in jobs),
        "craft": sum(j.kind is JobKind.CRAFT for j in jobs),
        "dependent": sum(bool(j.depends_on) for j in jobs),
    }


# --------------------------------------------------------------------------- main entry


def _first_valid_ordering(
    candidate: Scenario,
    triggers: list[int],
    disruption_types: list[DisruptionType] | tuple[DisruptionType, ...],
) -> tuple[tuple[DisruptionSpec, ...], list, int | None] | None:
    """Find an assignment of the scenario's disruption types to its trigger points such that, on
    the reference (optimal, re-planning) trajectory, every disruption fires and raises the optimal
    cost (FP2 section 4.4). Orderings are tried starting with the given one; only the order changes,
    so the multiset of types (and the cell's type balance) is preserved. Returns None if no
    ordering works. Otherwise returns (schedule, replay events, reference write actions)."""
    from .disruptions import all_fired_strictly, replay_optimal  # local: avoids import cycle

    if not disruption_types:
        return (), [], None
    for order in dict.fromkeys(itertools.permutations(disruption_types)):
        schedule = tuple(DisruptionSpec(i, t, p) for i, (t, p) in enumerate(zip(order, triggers)))
        events, steps = replay_optimal(dataclasses.replace(candidate, schedule=schedule))
        if all_fired_strictly(events):
            return schedule, events, steps
    return None


def generate_scenario(
    m: MapSpec,
    *,
    scenario_id: str,
    set_name: str,
    length_level: int,
    disruption_types: list[DisruptionType] | tuple[DisruptionType, ...],
    seed: int,
    index: int = 0,
    config: GeneratorConfig | None = None,
) -> Scenario:
    """Sample until a scenario with a proven optimum inside the length band is found.

    ``index`` is the scenario's position within its length level; it fixes the job composition
    (``GeneratorConfig.planned_composition``)."""
    cfg = config or GeneratorConfig()
    if length_level not in cfg.jobs_per_level:
        raise ValueError(f"unknown length level {length_level}")
    lo, hi = cfg.band(length_level)
    n_craft, n_dependent = cfg.planned_composition(length_level, index)
    rejections: dict[str, int] = {}
    solve_seconds = 0.0

    for attempt in range(cfg.max_attempts):
        rng = random.Random(seed * 1000 + attempt)
        try:
            start, items, jobs, radius = sample_candidate(
                m, length_level, rng, cfg, n_craft, n_dependent
            )
        except GenerationError:
            rejections["constraint"] = rejections.get("constraint", 0) + 1
            continue
        total = count_subgoals(jobs)
        try:
            triggers = trigger_points(total, len(disruption_types))
        except GenerationError:
            rejections["too_few_subgoals"] = rejections.get("too_few_subgoals", 0) + 1
            continue
        schedule = tuple(
            DisruptionSpec(i, t, p) for i, (t, p) in enumerate(zip(disruption_types, triggers))
        )
        candidate = Scenario(
            id=scenario_id, seed=seed, set_name=set_name, map=m, start_location=start,
            items=tuple(items), jobs=tuple(jobs), schedule=schedule, capacity=cfg.capacity,
            length_level=length_level, disruption_level=len(disruption_types),
            total_subgoals=total,
        )
        check_scenario(candidate)  # generator bug if this fails
        res = solve(World.from_scenario(candidate), ProgressTracker(candidate.jobs), cfg.solver)
        solve_seconds += res.stats.seconds
        if res.solvable is False:
            reason = "unsolvable"
        elif not res.proven_optimal:
            reason = "unproven"
        elif res.cost < lo:
            reason = "too_short"
        elif res.cost > hi:
            reason = "too_long"
        else:
            accepted = _first_valid_ordering(candidate, triggers, disruption_types)
            if accepted is None:
                reason = "schedule_replay"
                rejections[reason] = rejections.get(reason, 0) + 1
                continue
            schedule, replay_events, ref_steps = accepted
            return dataclasses.replace(
                candidate,
                schedule=schedule,
                optimal_actions=res.cost,
                reference_actions=res.cost if ref_steps is None else ref_steps,
                metadata={
                    "generator_version": GENERATOR_VERSION,
                    "attempt": attempt,
                    "rejections": dict(sorted(rejections.items())),
                    "radius": radius,
                    "composition": composition(jobs),
                    "solve_seconds_total": round(solve_seconds, 3),
                    "solve_expanded": res.stats.expanded,
                    "disruption_order_changed": [d.type for d in schedule] != list(disruption_types),
                    "reference_replay": [
                        {"index": e.index, "type": e.type.value, "target": e.target,
                         "cost_before": e.cost_before, "cost_after": e.cost_after,
                         "affected_jobs": e.affected_jobs}
                        for e in replay_events
                    ],
                },
            )
        rejections[reason] = rejections.get(reason, 0) + 1
    raise GenerationError(
        f"{scenario_id}: no scenario accepted in {cfg.max_attempts} attempts; rejections "
        f"{dict(sorted(rejections.items()))}"
    )
