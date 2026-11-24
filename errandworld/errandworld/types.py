"""Core data model for ErrandWorld.

These are immutable *specifications*: what a map or scenario looks like at the start of an
episode. They hold no simulation logic. The live, mutable world state lives in ``world.py``,
which copies from these specs.

Conventions
-----------
- All ids are plain strings and unique within their kind (locations, items, stations, jobs).
- Collections are tuples so specs are hashable and compare by value.
- Roads are undirected; ``Road.a < Road.b`` always holds (normalized on construction).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

# Special values for Item.location in the *live* world (world.py). A scenario's initial items
# always sit at a real location id.
AGENT = "agent"  # carried by the agent
CONSUMED = "consumed"  # used up by a station recipe

RESERVED_LOCATION_IDS = frozenset({AGENT, CONSUMED})


class JobKind(StrEnum):
    DELIVER = "deliver"  # bring an item of target_type to destination
    CRAFT = "craft"  # produce target_type at a station from its recipe, deliver it


class DisruptionType(StrEnum):
    ROAD_CLOSURE = "road_closure"
    MOVED_ITEM = "moved_item"
    STATION_OFFLINE = "station_offline"
    STALE_MAP = "stale_map"


@dataclass(frozen=True)
class Location:
    id: str
    name: str


@dataclass(frozen=True)
class Road:
    """Undirected road between two locations. Endpoints are stored sorted."""

    a: str
    b: str
    open: bool = True

    def __post_init__(self) -> None:
        if self.a == self.b:
            raise ValueError(f"road cannot be a self-loop: {self.a!r}")
        if self.a > self.b:
            a, b = self.b, self.a
            object.__setattr__(self, "a", a)
            object.__setattr__(self, "b", b)

    @property
    def key(self) -> tuple[str, str]:
        return (self.a, self.b)

    def other(self, loc: str) -> str:
        if loc == self.a:
            return self.b
        if loc == self.b:
            return self.a
        raise ValueError(f"{loc!r} is not an endpoint of road {self.key}")


def road_key(x: str, y: str) -> tuple[str, str]:
    """Normalized key for the road between x and y."""
    return (x, y) if x < y else (y, x)


@dataclass(frozen=True)
class Item:
    id: str
    type: str
    location: str  # a location id (in a scenario spec); may also be AGENT/CONSUMED in a live world


@dataclass(frozen=True)
class Station:
    id: str
    type: str
    location: str
    online: bool = True


@dataclass(frozen=True)
class Recipe:
    """A station of ``station_type`` turns carried ``inputs`` into one ``output`` item."""

    station_type: str
    inputs: tuple[str, ...]
    output: str

    def matches(self, station_type: str, input_types: list[str] | tuple[str, ...]) -> bool:
        """True if this recipe applies to these inputs (order-insensitive, multiset match)."""
        return station_type == self.station_type and sorted(input_types) == sorted(self.inputs)


@dataclass(frozen=True)
class Job:
    """One errand.

    - DELIVER: bring any item of ``target_type`` to ``destination``.
    - CRAFT: produce ``target_type`` via its recipe (see recipes.RECIPES), deliver to ``destination``.

    ``depends_on`` lists jobs whose delivered output this job consumes as a recipe input.
    """

    id: str
    kind: JobKind
    target_type: str
    destination: str
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class DisruptionSpec:
    """A scheduled disruption. It fires once ``trigger_subgoal`` subgoals have been completed.

    The concrete target (which road/item/station) is chosen at fire time by the injector,
    from the live state, so it is not part of the spec.
    """

    index: int
    type: DisruptionType
    trigger_subgoal: int


@dataclass(frozen=True)
class MapSpec:
    """Static map: locations, roads, and stations. Shared by many scenarios."""

    id: str
    seed: int
    locations: tuple[Location, ...]
    roads: tuple[Road, ...]
    stations: tuple[Station, ...]

    def location_ids(self) -> list[str]:
        return sorted(loc.id for loc in self.locations)


@dataclass(frozen=True)
class Scenario:
    """A fully specified, self-contained episode setup (map embedded)."""

    id: str
    seed: int
    set_name: str  # "dev" | "pilot" | "test" | free-form for hand-built scenarios
    map: MapSpec
    start_location: str
    items: tuple[Item, ...]
    jobs: tuple[Job, ...]
    schedule: tuple[DisruptionSpec, ...]
    capacity: int = 3
    length_level: int | None = None  # target optimal write actions: 10, 20, 40
    disruption_level: int | None = None  # scheduled disruptions: 0, 1, 3
    optimal_actions: int | None = None  # filled in by the generator after solving
    total_subgoals: int | None = None  # filled in by the generator
    # Optimal write actions *with* the scheduled disruptions, along the reference trajectory
    # (optimal play, re-planned optimally after each disruption). Equals optimal_actions when
    # there are no disruptions. The action cap and excess-action metric use this.
    reference_actions: int | None = None
    # Free-form notes (generator attempts, solver stats, ...). Excluded from equality and hashing.
    metadata: dict = field(default_factory=dict, compare=False, hash=False)
