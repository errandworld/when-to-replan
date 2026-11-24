"""Optimal planner for ErrandWorld: A* over write actions.

Cost model: every write action (move one road, pick, drop, use) costs 1; read tools cost 0. This
is the same count the action cap and the excess-actions metric use.

Abstraction
-----------
The search works on **item types**, not item ids. Every rule in ``World`` depends only on types
(recipes match types, deliveries match types), so items of the same type are interchangeable and
collapsing them removes symmetric states without changing the optimum. A search state is::

    (agent location, carried types (sorted), ground items as sorted (location, type) pairs,
     bitmask of completed jobs)

Only *relevant* ground items are tracked: types that some pending job delivers, consumes, or
produces. Irrelevant items on the ground can never help. Carried items are always tracked (they
use capacity). Stations' online status and road closures are fixed for the duration of a search.
The solver reads the **true** world state; the agent's stale map view plays no role.

Actions considered: move to an open neighbour; pick a relevant type here (only while carrying
fewer than the pending jobs could use); drop any carried type here (a delivery if it completes a
pending job, otherwise a staging drop); use an online station here whose output is still useful.
These prunings never remove an optimal plan (an optimal plan never holds more copies of a type
than the remaining jobs can use, and never crafts an unusable output).

Partial-order reduction: if a delivery is available, or a *safe* station use (no input is a
pending delivery target or needed by another pending job), it is the only successor. Doing it
now is never worse, and this removes equivalent reorderings of local actions.

Heuristic (admissible: a sum of lower bounds on disjoint action kinds)
---------------------------------------------------------------------
Possession argument: each delivery or station use *ends* one holding of an item, and each holding
*starts* with a pick, a station use, or the item being carried now. Completion is latched, so one
item may be delivered, re-picked, and delivered again; the bounds therefore count actions, never
items.

- drops >= number of pending jobs (each completion is a separate drop);
- uses  >= number of crafted types a pending job delivers with no copy existing anywhere;
- picks >= per base type, (pending deliveries + required uses consuming it) minus copies carried;
- moves >= the shortest walk from the agent through every location it *must* visit (pending
  destinations; the only location holding a needed base type when none is carried; the only
  location with a usable station of a needed type), ignoring pick-before-drop order. This is
  looked up in a table precomputed per solve by bitmask DP over the candidate locations (at most
  ``MAX_ROUTE_NODES``); otherwise an MST bound over the same set is used.

If a required item type or location is unreachable, the state is dead (h = infinity).

Budgets and bounds
------------------
A* runs with an expansion and time budget. If the budget is hit, the f-value of the cheapest open
node is a proven ``lower_bound``; the solver then retries with weighted A* (``fallback_weight``)
for an ``upper_bound`` plan, and the result is ``proven_optimal`` only if the two meet. If no plan
is found, ``cost=None`` and ``solvable=None`` (unknown). Exhausting the search space without
reaching the goal proves the task unsolvable (``solvable=False``).

Measured limits (scripts/bench_solver.py, 15-location maps): exact solves are fast up to about 6
pending jobs, but 12-job tasks with craft jobs usually exceed a 10 s budget, independent of
inventory capacity (2, 3, 4 and 6 were tried).
"""

from __future__ import annotations

import heapq
import math
import itertools
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from .progress import ProgressTracker
from .recipes import CRAFTED_TYPES, RECIPES
from .types import AGENT, Job, JobKind
from .world import World

INF = float("inf")

# Abstract actions: ("move", location_id) | ("pick", type) | ("drop", type) | ("use", output_type)
AbstractAction = tuple[str, str]


@dataclass(frozen=True)
class SolverConfig:
    max_expansions: int = 400_000
    # Wall-clock limit per search attempt. Defaults to no limit: budgets that depend on machine
    # load would make results (e.g. disruption targets) differ between runs. Benchmarks may set it.
    time_limit: float = float("inf")
    fallback_weight: float | None = 1.5  # None disables the weighted-A* fallback


@dataclass
class SolveStats:
    expanded: int = 0
    generated: int = 0
    seconds: float = 0.0
    weight: float = 1.0
    attempts: int = 0
    budget_hit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class SolveResult:
    """Outcome of a solve.

    ``cost`` is the length of the best plan found (``actions``). ``lower_bound`` and
    ``upper_bound`` bracket the true optimum: when the exact search finishes they both equal
    ``cost``; when it runs out of budget, ``lower_bound`` is the smallest f-value left on the
    open list (a proven bound for admissible A*) and ``upper_bound`` is the fallback plan's
    length. ``proven_optimal`` is true exactly when the bracket is closed.
    """

    cost: int | None
    actions: list[AbstractAction] | None
    proven_optimal: bool
    solvable: bool | None  # None = unknown (budget exhausted)
    stats: SolveStats = field(default_factory=SolveStats)
    lower_bound: int | None = None
    upper_bound: int | None = None

    @property
    def gap(self) -> int | None:
        if self.lower_bound is None or self.upper_bound is None:
            return None
        return self.upper_bound - self.lower_bound


# --------------------------------------------------------------------------- problem encoding


class _Problem:
    """Integer-encoded, search-ready view of a world plus pending jobs."""

    def __init__(self, world: World, jobs: tuple[Job, ...], done: tuple[bool, ...]) -> None:
        self.world = world
        self.jobs = jobs
        self.n_jobs = len(jobs)
        self.all_mask = (1 << self.n_jobs) - 1
        self.start_done = sum(1 << i for i, d in enumerate(done) if d)

        locs = sorted(world.location_names)
        self.loc_ids = locs
        self.loc_index = {l: i for i, l in enumerate(locs)}
        n = len(locs)
        self.nbrs: list[tuple[int, ...]] = [
            tuple(self.loc_index[x] for x in world.open_neighbors(l)) for l in locs
        ]
        self.dist = [self._bfs(i) for i in range(n)]

        # Masks of jobs (by index) that deliver / consume / target each type.
        self.deliver_mask: dict[str, int] = {}  # job delivers this type (any kind)
        self.consume_mask: dict[str, int] = {}  # craft job uses this type as an input
        self.craft_target_mask: dict[str, int] = {}  # craft job produces this type
        for i, j in enumerate(jobs):
            bit = 1 << i
            self.deliver_mask[j.target_type] = self.deliver_mask.get(j.target_type, 0) | bit
            if j.kind is JobKind.CRAFT:
                self.craft_target_mask[j.target_type] = (
                    self.craft_target_mask.get(j.target_type, 0) | bit
                )
                for t in RECIPES[j.target_type].inputs:
                    self.consume_mask[t] = self.consume_mask.get(t, 0) | bit
        self.relevant = (
            set(self.deliver_mask) | set(self.consume_mask) | set(self.craft_target_mask)
        )
        # Delivery lookup: (location index, type) -> job indices in job order.
        self.delivery_jobs: dict[tuple[int, str], list[int]] = {}
        for i, j in enumerate(jobs):
            self.delivery_jobs.setdefault((self.loc_index[j.destination], j.target_type), []).append(i)
        # Usable stations: location index -> station types online there.
        self.station_types_at: list[tuple[str, ...]] = [() for _ in locs]
        st_locs: dict[str, set[int]] = {}
        for sid in sorted(world.station_type):
            if world.station_online[sid]:
                li = self.loc_index[world.station_location[sid]]
                st = world.station_type[sid]
                st_locs.setdefault(st, set()).add(li)
        for st, ls in st_locs.items():
            for li in ls:
                self.station_types_at[li] = tuple(sorted(set(self.station_types_at[li]) | {st}))
        self.station_locs = {st: frozenset(ls) for st, ls in st_locs.items()}
        self.recipes_by_station: dict[str, list[str]] = {}
        for out in sorted(RECIPES):
            r = RECIPES[out]
            if out in self.craft_target_mask:  # only recipes some job needs
                self.recipes_by_station.setdefault(r.station_type, []).append(out)
        self._mst_cache: dict[tuple[int, frozenset[int]], int] = {}
        self._route_cache: dict[tuple[int, frozenset[int]], float] = {}
        self._init_route_table()

    # Largest set of candidate locations for which the exact route table is built (2^k * k^2
    # work, once per solve). Beyond this, the MST bound is used instead.
    MAX_ROUTE_NODES = 12

    def _init_route_table(self) -> None:
        """Precompute, by bitmask DP, the shortest walk that starts at node v and visits every
        node of S, for all subsets S of the candidate locations U. U holds every location the
        agent could be required to visit: destinations, locations of relevant items, and
        locations of usable stations. Any state's must-visit set is then a subset of U, unless a
        staging drop puts an item somewhere new (then the MST bound is used)."""
        u: set[int] = {self.loc_index[j.destination] for j in self.jobs}
        for item, loc in self.world.item_location.items():
            if loc in self.loc_index and self.world.item_type[item] in self.relevant:
                u.add(self.loc_index[loc])
        for out in self.craft_target_mask:
            u |= self.station_locs.get(RECIPES[out].station_type, frozenset())
        nodes = sorted(u)
        self.route_nodes = nodes
        self.route_bit = {loc: 1 << i for i, loc in enumerate(nodes)}
        self.route_table: list[list[float]] | None = None
        k = len(nodes)
        if k > self.MAX_ROUTE_NODES or k == 0:
            return
        d = [[self.dist[a][b] for b in nodes] for a in nodes]
        full = 1 << k
        # p[S][v]: shortest walk starting at node v (v in S) that visits all of S.
        p: list[list[float]] = [[INF] * k for _ in range(full)]
        for v in range(k):
            p[1 << v][v] = 0
        for s in range(1, full):
            row = p[s]
            for v in range(k):
                pv = row[v]
                if pv == INF:
                    continue
                for w in range(k):
                    bit = 1 << w
                    if s & bit:
                        continue
                    c = d[w][v] + pv  # new start w, then walk to v and continue
                    t = p[s | bit]
                    if c < t[w]:
                        t[w] = c
        self.route_table = p

    def _route(self, loc: int, nodes: frozenset[int]) -> float:
        """Lower bound on the moves needed to visit every node in ``nodes`` from ``loc``:
        the exact shortest covering walk (ignoring pick-before-drop order), or the MST bound
        when the table does not cover these nodes."""
        if not nodes or nodes == {loc}:
            return 0
        key = (loc, nodes)
        cached = self._route_cache.get(key)
        if cached is not None:
            return cached
        result: float
        if self.route_table is not None and all(n in self.route_bit for n in nodes):
            mask = 0
            for n in nodes:
                mask |= self.route_bit[n]
            row = self.route_table[mask]
            dl = self.dist[loc]
            result = min(
                (dl[n] + row[i] for i, n in enumerate(self.route_nodes) if mask >> i & 1),
                default=INF,
            )
        else:
            result = self._mst(loc, nodes)
        self._route_cache[key] = result
        return result

    def _bfs(self, src: int) -> list[float]:
        d = [INF] * len(self.loc_ids)
        d[src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            for v in self.nbrs[u]:
                if d[v] == INF:
                    d[v] = d[u] + 1
                    q.append(v)
        return d

    def start_state(self) -> tuple:
        w = self.world
        inv = tuple(sorted(w.item_type[i] for i in w.inventory))
        ground = []
        for item, loc in w.item_location.items():
            t = w.item_type[item]
            if loc in self.loc_index and t in self.relevant:
                ground.append((self.loc_index[loc], t))
        return (self.loc_index[w.agent_location], inv, tuple(sorted(ground)), self.start_done)

    # ------------------------------------------------------------------ counts

    @staticmethod
    def _pending(mask: int, done: int) -> int:
        return (mask & ~done).bit_count()

    def _max_needed(self, t: str, done: int) -> int:
        return self._pending(self.deliver_mask.get(t, 0) | self.consume_mask.get(t, 0), done)

    # ------------------------------------------------------------------ successors

    def successors(self, state: tuple):
        """Successor states. Applies partial-order reduction: if a provably safe local action
        exists (an available delivery, or a safe station use), it is the *only* successor, which
        removes the many equivalent orderings of local actions without losing an optimal plan.

        - Eager delivery: holding a pending job's item at its destination, delivering now is
          never worse (it frees a slot; the item stays here and can be re-picked).
        - Eager use: holding a recipe's inputs at an online station whose output is still needed,
          crafting now is never worse *provided* no pending job delivers an input type and no
          other pending job consumes it (so crafting cannot destroy something needed elsewhere).
        """
        loc, inv, ground, done = state

        for t in sorted(set(inv)):
            for ji in self.delivery_jobs.get((loc, t), ()):
                if not done >> ji & 1:
                    new_inv = list(inv)
                    new_inv.remove(t)
                    new_ground = tuple(sorted(ground + ((loc, t),)))
                    yield ("drop", t), (loc, tuple(new_inv), new_ground, done | (1 << ji))
                    return

        for act, nxt in self._uses(state):
            out = act[1]
            if all(
                not self._pending(self.deliver_mask.get(t, 0), done)
                and self._pending(self.consume_mask.get(t, 0), done) <= 1
                for t in RECIPES[out].inputs
            ):
                yield act, nxt
                return

        for nb in self.nbrs[loc]:
            yield ("move", self.loc_ids[nb]), (nb, inv, ground, done)

        here = [t for (l, t) in ground if l == loc]
        if here and len(inv) < self.world.capacity:
            for t in sorted(set(here)):
                if inv.count(t) >= self._max_needed(t, done):
                    continue
                g = list(ground)
                g.remove((loc, t))
                yield ("pick", t), (loc, tuple(sorted(inv + (t,))), tuple(g), done)

        # Staging drops (no delivery is possible here, or we would have returned above).
        for t in sorted(set(inv)):
            new_inv = list(inv)
            new_inv.remove(t)
            new_ground = (
                tuple(sorted(ground + ((loc, t),))) if t in self.relevant else ground
            )
            yield ("drop", t), (loc, tuple(new_inv), new_ground, done)

        yield from self._uses(state)

    def _uses(self, state: tuple):
        loc, inv, ground, done = state
        for st in self.station_types_at[loc]:
            for out in self.recipes_by_station.get(st, ()):
                r = RECIPES[out]
                remaining = list(inv)
                try:
                    for t in r.inputs:
                        remaining.remove(t)
                except ValueError:
                    continue
                existing = inv.count(out) + sum(1 for (_, t) in ground if t == out)
                useful = self._pending(self.craft_target_mask.get(out, 0), done) + self._pending(
                    self.consume_mask.get(out, 0), done
                )
                if existing >= useful:
                    continue
                yield ("use", out), (loc, tuple(sorted(remaining + [out])), ground, done)

    # ------------------------------------------------------------------ heuristic

    def heuristic(self, state: tuple) -> float:
        loc, inv, ground, done = state
        pending = self.all_mask & ~done
        if not pending:
            return 0
        h = pending.bit_count()  # drops
        must_visit: set[int] = set()
        for i in range(self.n_jobs):
            if pending >> i & 1:
                must_visit.add(self.loc_index[self.jobs[i].destination])

        ground_count: dict[str, int] = {}
        ground_locs: dict[str, set[int]] = {}
        for l, t in ground:
            ground_count[t] = ground_count.get(t, 0) + 1
            ground_locs.setdefault(t, set()).add(l)

        # Possession argument: every delivery drop or station use *ends* one holding of an item,
        # and every holding *starts* with a pick, a station use (crafted types), or being carried
        # now. Because completion is latched, one physical item may be delivered, re-picked, and
        # delivered again, so we bound action counts, never item counts.

        # uses: a crafted type that some pending job delivers, with no copy anywhere, needs >= 1
        # station use (2 jobs targeting the same type may share one copy via re-picking).
        required_uses: list[str] = []
        for out, mask in self.craft_target_mask.items():
            if (
                self._pending(mask, done)
                and inv.count(out) == 0
                and ground_count.get(out, 0) == 0
            ):
                required_uses.append(out)
                h += 1
                st_locs = self.station_locs.get(RECIPES[out].station_type, frozenset())
                if not any(self.dist[loc][l] < INF for l in st_locs):
                    return INF
                if len(st_locs) == 1:
                    must_visit |= st_locs

        # picks (base types only; crafted types may come straight from a station)
        demand: dict[str, int] = {}
        for t, mask in self.deliver_mask.items():
            if t not in CRAFTED_TYPES:
                demand[t] = demand.get(t, 0) + self._pending(mask, done)
        for out in required_uses:
            for t in RECIPES[out].inputs:
                if t not in CRAFTED_TYPES:
                    demand[t] = demand.get(t, 0) + 1
        for t, d in demand.items():
            carried = inv.count(t)
            picks = d - carried
            if picks <= 0:
                continue
            h += picks
            if carried == 0:
                # The next holding must start from an item lying on the ground right now.
                locs_t = ground_locs.get(t)
                if not locs_t or all(self.dist[loc][l] == INF for l in locs_t):
                    return INF  # base items cannot be created
                if len(locs_t) == 1:
                    must_visit |= locs_t

        moves = self._route(loc, frozenset(must_visit))
        return INF if moves == INF else h + moves

    def _mst(self, loc: int, nodes: frozenset[int]) -> float:
        key = (loc, nodes)
        cached = self._mst_cache.get(key)
        if cached is not None:
            return cached
        pts = [loc] + sorted(nodes - {loc})
        if len(pts) == 1:
            self._mst_cache[key] = 0
            return 0
        in_tree = [False] * len(pts)
        best = [INF] * len(pts)
        best[0] = 0
        total = 0
        for _ in range(len(pts)):
            u = min((i for i in range(len(pts)) if not in_tree[i]), key=lambda i: best[i])
            if best[u] == INF:
                self._mst_cache[key] = INF
                return INF
            in_tree[u] = True
            total += best[u]
            du = self.dist[pts[u]]
            for v in range(len(pts)):
                if not in_tree[v] and du[pts[v]] < best[v]:
                    best[v] = du[pts[v]]
        self._mst_cache[key] = total
        return total

    def is_goal(self, state: tuple) -> bool:
        return state[3] == self.all_mask


# --------------------------------------------------------------------------- search


def _astar(problem: _Problem, weight: float, config: SolverConfig, stats: SolveStats):
    """Returns (cost, actions) on success, ("unsolvable", None) if the space is exhausted, or
    ("budget", lower_bound) when the budget runs out. ``lower_bound`` is the f-value of the
    cheapest open node, a valid bound on the optimum only when ``weight == 1``."""
    start = problem.start_state()
    h0 = problem.heuristic(start)
    if h0 == INF:
        return "unsolvable", None
    counter = itertools.count()
    open_heap = [(weight * h0, h0, next(counter), 0, start)]
    best_g = {start: 0}
    parent: dict[tuple, tuple[tuple, AbstractAction] | None] = {start: None}
    deadline = time.perf_counter() + config.time_limit
    expanded = 0
    while open_heap:
        f, _, _, g, state = heapq.heappop(open_heap)
        if g > best_g.get(state, INF):
            continue  # stale entry
        if problem.is_goal(state):
            stats.expanded += expanded
            actions: list[AbstractAction] = []
            cur = state
            while parent[cur] is not None:
                prev, act = parent[cur]
                actions.append(act)
                cur = prev
            actions.reverse()
            return g, actions
        expanded += 1
        if expanded > config.max_expansions or (
            expanded % 2048 == 0 and time.perf_counter() > deadline
        ):
            stats.expanded += expanded
            stats.budget_hit = True
            # ``state`` was the cheapest open node, so its f bounds the optimum from below.
            return "budget", int(math.ceil(f - 1e-9)) if weight == 1.0 else None
        ng = g + 1
        for act, nxt in problem.successors(state):
            if ng >= best_g.get(nxt, INF):
                continue
            h = problem.heuristic(nxt)
            if h == INF:
                continue
            best_g[nxt] = ng
            parent[nxt] = (state, act)
            stats.generated += 1
            heapq.heappush(open_heap, (ng + weight * h, h, next(counter), ng, nxt))
    stats.expanded += expanded
    return "unsolvable", None


def solve(
    world: World, tracker: ProgressTracker, config: SolverConfig | None = None
) -> SolveResult:
    """Optimal remaining write actions to finish every pending job from the current state."""
    config = config or SolverConfig()
    jobs = tracker.jobs
    done = tuple(tracker.job_done(j.id) for j in jobs)
    stats = SolveStats()
    t0 = time.perf_counter()
    problem = _Problem(world, jobs, done)

    weights = [1.0] + ([config.fallback_weight] if config.fallback_weight else [])
    lower: int | None = None
    for w in weights:
        stats.attempts += 1
        stats.weight = w
        cost, actions = _astar(problem, w, config, stats)
        if cost == "unsolvable":
            stats.seconds = time.perf_counter() - t0
            # Exhaustion proves unsolvability (the prunings preserve all optimal plans, and
            # weighted A* also explores the full reachable space before giving up).
            return SolveResult(None, None, True, False, stats)
        if cost == "budget":
            if w == 1.0:
                lower = actions  # the proven lower bound
            continue
        stats.seconds = time.perf_counter() - t0
        lb = cost if w == 1.0 else lower
        return SolveResult(cost, actions, lb == cost, True, stats, lb, cost)
    stats.seconds = time.perf_counter() - t0
    return SolveResult(None, None, False, None, stats, lower, None)


def optimal_cost(world: World, tracker: ProgressTracker, config: SolverConfig | None = None):
    return solve(world, tracker, config).cost


def is_solvable(world: World, tracker: ProgressTracker, config: SolverConfig | None = None):
    """True / False, or None if the budget ran out before deciding."""
    return solve(world, tracker, config).solvable


# --------------------------------------------------------------------------- concrete plans


def concretize(world: World, actions: list[AbstractAction]) -> list[tuple[str, dict[str, Any]]]:
    """Turn abstract actions into concrete tool calls by replaying them on a clone.

    Picks the lowest-sorted item id of the needed type, and the first matching station. Raises
    AssertionError if any step fails (that would be a solver bug).
    """
    w = world.clone()
    calls: list[tuple[str, dict[str, Any]]] = []
    for kind, arg in actions:
        if kind == "move":
            call = ("move", {"location": arg})
            o = w.move(arg)
        elif kind == "pick":
            item = next(i for i in w.items_at(w.agent_location) if w.item_type[i] == arg)
            call = ("pick", {"item": item})
            o = w.pick(item)
        elif kind == "drop":
            item = next(i for i in w.inventory if w.item_type[i] == arg)
            call = ("drop", {"item": item})
            o = w.drop(item)
        elif kind == "use":
            r = RECIPES[arg]
            station = next(
                s for s in w.stations_at(w.agent_location)
                if w.station_type[s] == r.station_type and w.station_online[s]
            )
            inv = list(w.inventory)
            items = []
            for t in r.inputs:
                i = next(x for x in inv if w.item_type[x] == t)
                inv.remove(i)
                items.append(i)
            call = ("use", {"station": station, "items": items})
            o = w.use(station, items)
        else:  # pragma: no cover
            raise ValueError(kind)
        assert o.ok, (call, o)
        calls.append(call)
    return calls


def plan(
    world: World, tracker: ProgressTracker, config: SolverConfig | None = None
) -> list[tuple[str, dict[str, Any]]] | None:
    """An optimal (or fallback) sequence of concrete tool calls, or None if none was found."""
    res = solve(world, tracker, config)
    if res.actions is None:
        return None
    return concretize(world, res.actions)


__all__ = [
    "AGENT",
    "SolveResult",
    "SolveStats",
    "SolverConfig",
    "concretize",
    "is_solvable",
    "optimal_cost",
    "plan",
    "solve",
]
