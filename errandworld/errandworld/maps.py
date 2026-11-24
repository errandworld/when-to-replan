"""Map generation and graph utilities.

Map shape
---------
Each map is a **ring with chords**: the locations are arranged in a random cycle, then extra
roads ("chords") are added between random non-adjacent locations until the target average degree
is reached. This gives three properties the experiment needs:

1. **No bridges.** A cycle is 2-edge-connected, and adding edges keeps it so. Closing any single
   road leaves the map connected, so every road is a legal road-closure target and the task stays
   solvable (the injector still re-checks with the solver).
2. **Long trips on a small map.** A sparse ring has a large diameter, so task length can come from
   travel rather than from the number of jobs (design decision D6).
3. **Controlled detours.** Chords create alternative routes, so a closure usually forces a detour
   rather than a dead end.

Stations: every station type gets ``stations_per_type`` (default 2) stations at distinct
locations, at most one station per location, so taking one offline always leaves an alternative.

Determinism: everything is driven by ``random.Random(seed)`` and sorted iteration.
"""

from __future__ import annotations

import random
from collections import deque

from .recipes import STATION_TYPES
from .types import Location, MapSpec, Road, Station

# Neutral place names (no hint about what is found there). Ids are the snake_case form.
PLACE_NAMES: tuple[str, ...] = (
    "Ashford", "Birchwood", "Clearwater", "Dunmore", "Eastbrook", "Fairview", "Glenhaven",
    "Highfield", "Ironbridge", "Juniper", "Kingsley", "Lakeside", "Millbrook", "Northgate",
    "Oakridge", "Pinecrest", "Queensway", "Riverton", "Stonehill", "Thornbury", "Upton",
    "Westvale", "Yarrow", "Zephyr",
)


def place_id(name: str) -> str:
    return name.lower()


def generate_map(
    map_id: str,
    seed: int,
    *,
    n_locations: int = 15,
    avg_degree: float = 2.4,
    stations_per_type: int = 2,
) -> MapSpec:
    """Generate a ring-with-chords map. Raises ValueError for infeasible parameters."""
    if not 3 <= n_locations <= len(PLACE_NAMES):
        raise ValueError(f"n_locations must be in [3, {len(PLACE_NAMES)}]")
    if avg_degree < 2.0:
        raise ValueError("avg_degree must be >= 2 (the ring alone has degree 2)")
    n_stations = stations_per_type * len(STATION_TYPES)
    if n_stations > n_locations:
        raise ValueError("more stations than locations (at most one station per location)")

    rng = random.Random(seed)
    names = rng.sample(PLACE_NAMES, n_locations)
    ids = [place_id(n) for n in names]

    ring = list(ids)
    rng.shuffle(ring)
    edges: set[tuple[str, str]] = set()
    for i in range(n_locations):
        a, b = ring[i], ring[(i + 1) % n_locations]
        edges.add((a, b) if a < b else (b, a))

    target_edges = round(avg_degree * n_locations / 2)
    max_edges = n_locations * (n_locations - 1) // 2
    target_edges = min(target_edges, max_edges)
    candidates = sorted(
        (a, b) for i, a in enumerate(sorted(ids)) for b in sorted(ids)[i + 1:] if (a, b) not in edges
    )
    rng.shuffle(candidates)
    while len(edges) < target_edges and candidates:
        edges.add(candidates.pop())

    station_locs = rng.sample(sorted(ids), n_stations)
    stations = []
    k = 0
    for st in STATION_TYPES:
        for n in range(1, stations_per_type + 1):
            stations.append(Station(f"{st}_{n}", st, station_locs[k]))
            k += 1

    return MapSpec(
        id=map_id,
        seed=seed,
        locations=tuple(Location(place_id(n), n) for n in sorted(names)),
        roads=tuple(Road(a, b) for a, b in sorted(edges)),
        stations=tuple(stations),
    )


# --------------------------------------------------------------------------- graph utilities


def adjacency(m: MapSpec, *, open_only: bool = True) -> dict[str, list[str]]:
    adj: dict[str, list[str]] = {l.id: [] for l in m.locations}
    for r in m.roads:
        if r.open or not open_only:
            adj[r.a].append(r.b)
            adj[r.b].append(r.a)
    for v in adj.values():
        v.sort()
    return adj


def distances_from(m: MapSpec, src: str) -> dict[str, int]:
    adj = adjacency(m)
    dist = {src: 0}
    q = deque([src])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def is_connected(m: MapSpec) -> bool:
    if not m.locations:
        return True
    return len(distances_from(m, m.locations[0].id)) == len(m.locations)


def diameter(m: MapSpec) -> int | None:
    """Longest shortest path, or None if disconnected."""
    best = 0
    for l in m.locations:
        d = distances_from(m, l.id)
        if len(d) < len(m.locations):
            return None
        best = max(best, max(d.values()))
    return best


def bridges_of(adj: dict[str, list[str]]) -> list[tuple[str, str]]:
    """Bridges of an undirected graph given as an adjacency dict, as sorted (a, b) keys."""
    disc: dict[str, int] = {}
    low: dict[str, int] = {}
    out: list[tuple[str, str]] = []
    counter = [0]

    def dfs(u: str, parent: str | None) -> None:
        disc[u] = low[u] = counter[0]
        counter[0] += 1
        skipped_parent = False
        for v in adj[u]:
            if v == parent and not skipped_parent:
                skipped_parent = True  # skip the tree edge back to the parent once
                continue
            if v in disc:
                low[u] = min(low[u], disc[v])
            else:
                dfs(v, u)
                low[u] = min(low[u], low[v])
                if low[v] > disc[u]:
                    out.append((u, v) if u < v else (v, u))

    for l in sorted(adj):
        if l not in disc:
            dfs(l, None)
    return sorted(out)


def bridges(m: MapSpec) -> list[tuple[str, str]]:
    """Roads whose removal disconnects the (open-road) graph, as sorted (a, b) keys."""
    return bridges_of(adjacency(m))


def map_invariant_problems(m: MapSpec, *, stations_per_type: int = 2) -> list[str]:
    """Experiment-level invariants on top of scenario_io.map_problems (schema-level)."""
    p: list[str] = []
    if not is_connected(m):
        p.append("map is not connected")
    br = bridges(m)
    if br:
        p.append(f"map has bridges: {br}")
    for st in STATION_TYPES:
        locs = [s.location for s in m.stations if s.type == st]
        if len(set(locs)) < stations_per_type:
            p.append(f"station type {st!r} has {len(set(locs))} distinct locations, "
                     f"need {stations_per_type}")
    st_locs = [s.location for s in m.stations]
    if len(st_locs) != len(set(st_locs)):
        p.append("more than one station at a location")
    return p


def describe(m: MapSpec) -> dict:
    n = len(m.locations)
    return {
        "id": m.id,
        "locations": n,
        "roads": len(m.roads),
        "avg_degree": round(2 * len(m.roads) / n, 2) if n else 0,
        "diameter": diameter(m),
        "bridges": len(bridges(m)),
        "stations": len(m.stations),
    }
