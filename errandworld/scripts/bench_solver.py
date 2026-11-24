"""Solver timing benchmark (Phase 1, step 6 gate).

Builds random scenarios on 15-location maps (2 stations per type) and times the optimal solver.
Maps come from errandworld.maps (ring with chords). Scenarios use a minimal stand-in builder;
the real generator arrives in step 8.

    python scripts/bench_solver.py --jobs 12 --craft-frac 0.5 --radius 2 --n 30
    python scripts/bench_solver.py --sweep            # the standard grid used in the report

Columns: optimal cost, expansions, seconds, and how often the budget was hit.
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from errandworld.maps import generate_map  # noqa: E402
from errandworld.progress import ProgressTracker, count_subgoals  # noqa: E402
from errandworld.recipes import BASE_ITEM_TYPES, RECIPES, STATION_TYPES, crafted_input, tier  # noqa: E402
from errandworld.solver import SolverConfig, solve  # noqa: E402
from errandworld.types import Item, Job, JobKind, Location, MapSpec, Road, Scenario, Station  # noqa: E402
from errandworld.world import World  # noqa: E402


def random_map(rng: random.Random, n: int = 15, avg_degree: float = 3.0) -> MapSpec:
    locs = [f"loc_{i:02d}" for i in range(n)]
    edges = set()
    for i in range(1, n):
        edges.add(tuple(sorted((locs[i], locs[rng.randrange(i)]))))
    while 2 * len(edges) / n < avg_degree:
        a, b = rng.sample(locs, 2)
        edges.add(tuple(sorted((a, b))))
    stations = []
    for st in STATION_TYPES:
        for k, loc in enumerate(rng.sample(locs, 2), 1):
            stations.append(Station(f"{st}_{k}", st, loc))
    return MapSpec(
        "bench", 0, tuple(Location(l, l) for l in locs),
        tuple(Road(a, b) for a, b in sorted(edges)), tuple(stations),
    )


def ball(m: MapSpec, center: str, radius: int) -> list[str]:
    nbrs = {l.id: [] for l in m.locations}
    for r in m.roads:
        nbrs[r.a].append(r.b)
        nbrs[r.b].append(r.a)
    dist = {center: 0}
    q = deque([center])
    while q:
        u = q.popleft()
        for v in nbrs[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return sorted(l for l, d in dist.items() if d <= radius)


def random_scenario(rng: random.Random, n_jobs: int, craft_frac: float, radius: int,
                    dep_frac: float = 0.25, capacity: int = 3, degree: float = 2.4) -> Scenario:
    m = generate_map("bench", rng.randrange(10**9), avg_degree=degree)
    hub = rng.choice([l.id for l in m.locations])
    area = ball(m, hub, radius)
    jobs: list[Job] = []
    items: list[Item] = []
    counter = [0]
    used_deliver: set[str] = set()
    used_outputs: set[str] = set()

    def add(t):
        counter[0] += 1
        items.append(Item(f"{t}_{counter[0]}", t, rng.choice(area)))

    tier1 = sorted(o for o in RECIPES if tier(o) == 1)
    tier2 = sorted(o for o in RECIPES if tier(o) == 2)
    deliver_slots: list[str] = []
    for k in range(n_jobs):
        jid = f"job_{k + 1}"
        if rng.random() < craft_frac:
            producers = [j for j in jobs if j.target_type in tier1
                         and any(crafted_input(o) == j.target_type for o in tier2
                                 if o not in used_outputs)]
            if producers and rng.random() < dep_frac:
                p = rng.choice(producers)
                out = next(o for o in tier2 if crafted_input(o) == p.target_type)
                jobs.append(Job(jid, JobKind.CRAFT, out, rng.choice(area), (p.id,)))
                used_outputs.add(out)
                for t in RECIPES[out].inputs:
                    if t not in RECIPES:
                        add(t)
                continue
            free = [o for o in tier1 if o not in used_outputs]
            if free:
                out = rng.choice(free)
                used_outputs.add(out)
                jobs.append(Job(jid, JobKind.CRAFT, out, rng.choice(area)))
                for t in RECIPES[out].inputs:
                    add(t)
                continue
        deliver_slots.append(jid)
    # Generator rules (step 8): deliver targets are distinct and are never craft inputs in the
    # same scenario, so no item can be delivered and then re-used (the latching loophole).
    craft_inputs = {i for j in jobs if j.kind is JobKind.CRAFT for i in RECIPES[j.target_type].inputs}
    pool = [t for t in BASE_ITEM_TYPES if t not in craft_inputs]
    rng.shuffle(pool)
    for jid in deliver_slots:
        if not pool:
            break  # ran out of eligible types: the scenario has fewer jobs
        t = pool.pop()
        used_deliver.add(t)
        jobs.append(Job(jid, JobKind.DELIVER, t, rng.choice(area)))
        add(t)
    jobs.sort(key=lambda j: int(j.id.split("_")[1]))
    unused = [t for t in BASE_ITEM_TYPES if t not in craft_inputs | used_deliver]
    for _ in range(3):
        add(rng.choice(unused or list(BASE_ITEM_TYPES)))
    return Scenario(
        id="bench", seed=0, set_name="bench", map=m, start_location=hub, items=tuple(items),
        jobs=tuple(jobs), schedule=(), capacity=capacity, total_subgoals=count_subgoals(jobs),
    )


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def bench(n_jobs, craft_frac, radius, n, seed, config, capacity=3, degree=2.4):
    rng = random.Random(seed)
    rows, scen = [], []
    for _ in range(n):
        s = random_scenario(rng, n_jobs, craft_frac, radius, capacity=capacity, degree=degree)
        scen.append(s)
        w = World.from_scenario(s)
        res = solve(w, ProgressTracker(s.jobs), config)
        rows.append(res)
    secs = [r.stats.seconds for r in rows]
    exp = [r.stats.expanded for r in rows]
    costs = [r.cost for r in rows if r.cost is not None]
    proven = sum(r.proven_optimal and r.cost is not None for r in rows)
    njobs = [len(r_s.jobs) for r_s in scen]
    print(
        f"deg={degree} jobs actual={min(njobs)}-{max(njobs)} cap={capacity} jobs={n_jobs:2d} craft={craft_frac:.2f} radius={radius} n={n}: "
        f"cost median={statistics.median(costs) if costs else '-'} "
        f"[{min(costs) if costs else '-'}-{max(costs) if costs else '-'}]  "
        f"time median={statistics.median(secs):.3f}s p95={pct(secs, 95):.3f}s max={max(secs):.3f}s  "
        f"expanded median={int(statistics.median(exp))} max={max(exp)}  "
        f"proven optimal {proven}/{n}",
        flush=True,
    )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--craft-frac", type=float, default=0.5)
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-expansions", type=int, default=400_000)
    ap.add_argument("--time-limit", type=float, default=20.0)
    ap.add_argument("--capacity", type=int, default=3)
    ap.add_argument("--degree", type=float, default=2.4)
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()
    config = SolverConfig(a.max_expansions, a.time_limit, fallback_weight=None)
    if a.sweep:
        for jobs in (3, 6, 12):
            for craft in (0.0, 0.3, 0.6):
                for radius in (1, 2, 3):
                    bench(jobs, craft, radius, a.n, a.seed, config, a.capacity, a.degree)
    else:
        bench(a.jobs, a.craft_frac, a.radius, a.n, a.seed, config, a.capacity, a.degree)


if __name__ == "__main__":
    main()
