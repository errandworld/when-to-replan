"""JSON serialization and integrity checks for maps and scenarios.

Every file carries ``schema_version``. Loading validates referential integrity and raises
``ScenarioError`` listing every problem found. ``fingerprint`` gives a stable hash of a
scenario's content, recorded in run traces for reproducibility.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .recipes import CRAFTED_TYPES, RECIPES
from .types import (
    RESERVED_LOCATION_IDS,
    DisruptionSpec,
    DisruptionType,
    Item,
    Job,
    JobKind,
    Location,
    MapSpec,
    Road,
    Scenario,
    Station,
)

SCHEMA_VERSION = 1


class ScenarioError(ValueError):
    """A map or scenario failed validation or could not be parsed."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("; ".join(problems))


# --------------------------------------------------------------------------- to dict


def map_to_dict(m: MapSpec) -> dict[str, Any]:
    return {
        "id": m.id,
        "seed": m.seed,
        "locations": [{"id": l.id, "name": l.name} for l in m.locations],
        "roads": [{"a": r.a, "b": r.b, "open": r.open} for r in m.roads],
        "stations": [
            {"id": s.id, "type": s.type, "location": s.location, "online": s.online}
            for s in m.stations
        ],
    }


def scenario_to_dict(s: Scenario) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "id": s.id,
        "seed": s.seed,
        "set_name": s.set_name,
        "length_level": s.length_level,
        "disruption_level": s.disruption_level,
        "capacity": s.capacity,
        "start_location": s.start_location,
        "optimal_actions": s.optimal_actions,
        "total_subgoals": s.total_subgoals,
        "reference_actions": s.reference_actions,
        "map": map_to_dict(s.map),
        "items": [{"id": i.id, "type": i.type, "location": i.location} for i in s.items],
        "jobs": [
            {
                "id": j.id,
                "kind": j.kind.value,
                "target_type": j.target_type,
                "destination": j.destination,
                "depends_on": list(j.depends_on),
            }
            for j in s.jobs
        ],
        "schedule": [
            {"index": d.index, "type": d.type.value, "trigger_subgoal": d.trigger_subgoal}
            for d in s.schedule
        ],
        "metadata": s.metadata,
    }


# --------------------------------------------------------------------------- from dict


def map_from_dict(d: dict[str, Any]) -> MapSpec:
    return MapSpec(
        id=d["id"],
        seed=int(d["seed"]),
        locations=tuple(Location(x["id"], x["name"]) for x in d["locations"]),
        roads=tuple(Road(x["a"], x["b"], bool(x.get("open", True))) for x in d["roads"]),
        stations=tuple(
            Station(x["id"], x["type"], x["location"], bool(x.get("online", True)))
            for x in d["stations"]
        ),
    )


def scenario_from_dict(d: dict[str, Any], *, validate: bool = True) -> Scenario:
    version = d.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ScenarioError([f"unsupported schema_version {version!r} (expected {SCHEMA_VERSION})"])
    try:
        s = Scenario(
            id=d["id"],
            seed=int(d["seed"]),
            set_name=d["set_name"],
            map=map_from_dict(d["map"]),
            start_location=d["start_location"],
            items=tuple(Item(x["id"], x["type"], x["location"]) for x in d["items"]),
            jobs=tuple(
                Job(
                    id=x["id"],
                    kind=JobKind(x["kind"]),
                    target_type=x["target_type"],
                    destination=x["destination"],
                    depends_on=tuple(x.get("depends_on", ())),
                )
                for x in d["jobs"]
            ),
            schedule=tuple(
                DisruptionSpec(int(x["index"]), DisruptionType(x["type"]), int(x["trigger_subgoal"]))
                for x in d["schedule"]
            ),
            capacity=int(d.get("capacity", 3)),
            length_level=d.get("length_level"),
            disruption_level=d.get("disruption_level"),
            optimal_actions=d.get("optimal_actions"),
            total_subgoals=d.get("total_subgoals"),
            reference_actions=d.get("reference_actions"),
            metadata=dict(d.get("metadata", {})),
        )
    except (KeyError, TypeError, ValueError) as e:
        raise ScenarioError([f"malformed scenario: {type(e).__name__}: {e}"]) from e
    if validate:
        check_scenario(s)
    return s


# --------------------------------------------------------------------------- validation


def _duplicates(ids: list[str]) -> list[str]:
    seen: set[str] = set()
    dups: set[str] = set()
    for x in ids:
        (dups if x in seen else seen).add(x)
    return sorted(dups)


def map_problems(m: MapSpec) -> list[str]:
    p: list[str] = []
    loc_ids = [l.id for l in m.locations]
    locs = set(loc_ids)
    for dup in _duplicates(loc_ids):
        p.append(f"duplicate location id {dup!r}")
    for bad in sorted(locs & RESERVED_LOCATION_IDS):
        p.append(f"location id {bad!r} is reserved")
    for dup in _duplicates([f"{r.a}|{r.b}" for r in m.roads]):
        p.append(f"duplicate road {dup!r}")
    for r in m.roads:
        for end in (r.a, r.b):
            if end not in locs:
                p.append(f"road {r.key} references unknown location {end!r}")
    for dup in _duplicates([s.id for s in m.stations]):
        p.append(f"duplicate station id {dup!r}")
    for s in m.stations:
        if s.location not in locs:
            p.append(f"station {s.id!r} at unknown location {s.location!r}")
    return p


def scenario_problems(s: Scenario) -> list[str]:
    """All integrity problems in a scenario (empty list = valid)."""
    p = map_problems(s.map)
    locs = {l.id for l in s.map.locations}
    station_types = {st.type for st in s.map.stations}

    if s.start_location not in locs:
        p.append(f"start_location {s.start_location!r} is not a location")
    if s.capacity < 2:
        p.append(f"capacity {s.capacity} < 2 (craft recipes need two inputs)")

    # Items
    station_ids = {st.id for st in s.map.stations}
    for dup in _duplicates([i.id for i in s.items]):
        p.append(f"duplicate item id {dup!r}")
    for i in s.items:
        if i.location not in locs:
            p.append(f"item {i.id!r} at unknown location {i.location!r}")
        if i.id in station_ids or i.id in locs:
            p.append(f"item id {i.id!r} collides with a station or location id")

    # Jobs
    job_ids = [j.id for j in s.jobs]
    jobs = {j.id: j for j in s.jobs}
    for dup in _duplicates(job_ids):
        p.append(f"duplicate job id {dup!r}")
    if not s.jobs:
        p.append("scenario has no jobs")
    item_types = {i.type for i in s.items}
    for j in s.jobs:
        if j.destination not in locs:
            p.append(f"job {j.id!r} destination {j.destination!r} is not a location")
        for dep in j.depends_on:
            if dep not in jobs:
                p.append(f"job {j.id!r} depends on unknown job {dep!r}")
            elif dep == j.id:
                p.append(f"job {j.id!r} depends on itself")
        if j.kind is JobKind.CRAFT:
            recipe = RECIPES.get(j.target_type)
            if recipe is None:
                p.append(f"craft job {j.id!r}: no recipe produces {j.target_type!r}")
                continue
            if recipe.station_type not in station_types:
                p.append(f"craft job {j.id!r}: no {recipe.station_type!r} station on the map")
            dep_outputs = {jobs[d].target_type for d in j.depends_on if d in jobs}
            for inp in recipe.inputs:
                if inp in CRAFTED_TYPES:
                    if inp not in dep_outputs:
                        p.append(
                            f"craft job {j.id!r}: input {inp!r} is crafted but no depends_on job "
                            f"produces it"
                        )
                elif inp not in item_types:
                    p.append(f"craft job {j.id!r}: no item of input type {inp!r} on the map")
        else:  # DELIVER
            if j.target_type in CRAFTED_TYPES:
                p.append(f"deliver job {j.id!r}: target {j.target_type!r} is crafted; use CRAFT")
            elif j.target_type not in item_types:
                p.append(f"deliver job {j.id!r}: no item of type {j.target_type!r} on the map")
        # A dependency means "consumes the crafted item that job delivered", so the producer must
        # be a CRAFT job whose output is one of this job's recipe inputs.
        consumed = (
            set(RECIPES[j.target_type].inputs)
            if j.kind is JobKind.CRAFT and j.target_type in RECIPES
            else set()
        )
        for dep in j.depends_on:
            if dep in jobs and dep != j.id and not (
                jobs[dep].kind is JobKind.CRAFT and jobs[dep].target_type in consumed
            ):
                p.append(
                    f"job {j.id!r} depends on {dep!r} but does not consume its crafted output"
                )
    if _has_cycle(jobs):
        p.append("job dependencies contain a cycle")

    # Schedule
    for dup in _duplicates([str(d.index) for d in s.schedule]):
        p.append(f"duplicate disruption index {dup}")
    triggers = [d.trigger_subgoal for d in s.schedule]
    if triggers != sorted(triggers):
        p.append("disruption schedule is not ordered by trigger_subgoal")
    for d in s.schedule:
        if d.trigger_subgoal < 1:
            p.append(f"disruption {d.index}: trigger_subgoal must be >= 1")
        if s.total_subgoals is not None and d.trigger_subgoal >= s.total_subgoals:
            p.append(
                f"disruption {d.index}: trigger_subgoal {d.trigger_subgoal} must be < "
                f"total_subgoals {s.total_subgoals}"
            )
    if s.disruption_level is not None and s.disruption_level != len(s.schedule):
        p.append(
            f"disruption_level {s.disruption_level} != number of scheduled disruptions "
            f"{len(s.schedule)}"
        )
    return p


def _has_cycle(jobs: dict[str, Job]) -> bool:
    state: dict[str, int] = {}  # 1 = visiting, 2 = done

    def visit(j: str) -> bool:
        if state.get(j) == 1:
            return True
        if state.get(j) == 2 or j not in jobs:
            return False
        state[j] = 1
        if any(visit(d) for d in jobs[j].depends_on):
            return True
        state[j] = 2
        return False

    return any(visit(j) for j in sorted(jobs))


def check_scenario(s: Scenario) -> None:
    problems = scenario_problems(s)
    if problems:
        raise ScenarioError(problems)


def check_map(m: MapSpec) -> None:
    problems = map_problems(m)
    if problems:
        raise ScenarioError(problems)


# --------------------------------------------------------------------------- files & hashing


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fingerprint(s: Scenario) -> str:
    """Stable SHA-256 of a scenario's content (metadata excluded)."""
    d = scenario_to_dict(s)
    d.pop("metadata", None)
    return hashlib.sha256(canonical_json(d).encode("utf-8")).hexdigest()


def _write_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ScenarioError([f"{path}: invalid JSON: {e}"]) from e


def save_scenario(s: Scenario, path: str | Path) -> None:
    check_scenario(s)
    _write_json(path, scenario_to_dict(s))


def load_scenario(path: str | Path) -> Scenario:
    return scenario_from_dict(_read_json(path))


def save_map(m: MapSpec, path: str | Path) -> None:
    check_map(m)
    _write_json(path, {"schema_version": SCHEMA_VERSION, **map_to_dict(m)})


def load_map(path: str | Path) -> MapSpec:
    d = _read_json(path)
    if d.get("schema_version") != SCHEMA_VERSION:
        raise ScenarioError([f"unsupported schema_version {d.get('schema_version')!r}"])
    try:
        m = map_from_dict(d)
    except (KeyError, TypeError, ValueError) as e:
        raise ScenarioError([f"malformed map: {type(e).__name__}: {e}"]) from e
    check_map(m)
    return m
