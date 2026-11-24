"""Live, mutable ErrandWorld state and its transition rules.

``World`` is the single source of truth during an episode. It has three kinds of methods:

1. **Agent actions** (``move``, ``pick``, ``drop``, ``use``): apply a write action and return an
   ``Outcome``. They never raise on bad input; a failed action returns an ``ErrorCode`` and leaves
   the state unchanged.
2. **Observers** (``look``, ``status``, ``read_map``): return structured dicts for the tool layer.
   ``look`` and ``status`` always show the true state; ``read_map`` shows the agent's map view,
   which may be stale (see below).
3. **Harness-only methods** (``close_road``, ``set_station_online``, ``move_item``, ``clone``,
   ``snapshot``, queries): used by the injector, solver, and evaluator. These raise ``KeyError`` /
   ``ValueError`` on misuse, since misuse is a harness bug, not agent behaviour.

Map view (design decision D5)
-----------------------------
``read_map`` reports the true state with a set of *stale overrides* applied. An override maps an
item id to the location the map still (wrongly) lists it at. Only a ``move_item(...,
update_map=False)`` call (the stale-map disruption) creates one. All other changes — road
closures, offline stations, moved items, and the agent's own actions — show up on the map
immediately. An override is cleared when the agent picks the item up (it now knows where it is) or
when the harness moves the item with ``update_map=True``.

Job progress is deliberately *not* tracked here; see ``progress.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .recipes import find_recipe
from .scenario_io import check_scenario
from .types import AGENT, CONSUMED, Scenario, road_key


class ErrorCode(StrEnum):
    NOT_ADJACENT = "not_adjacent"
    ROAD_CLOSED = "road_closed"
    ITEM_NOT_HERE = "item_not_here"
    INVENTORY_FULL = "inventory_full"
    NOT_CARRYING = "not_carrying"
    NOT_AT_STATION = "not_at_station"
    STATION_OFFLINE = "station_offline"
    MISSING_ITEMS = "missing_items"
    INVALID_RECIPE = "invalid_recipe"
    UNKNOWN_ID = "unknown_id"
    # Raised by the tool layer (tools.dispatch), never by World itself:
    UNKNOWN_TOOL = "unknown_tool"
    INVALID_ARGUMENTS = "invalid_arguments"
    # Raised by the episode layer (episode.Episode) once the episode has ended:
    EPISODE_OVER = "episode_over"


WORLD_ERROR_CODES = frozenset(ErrorCode) - {
    ErrorCode.UNKNOWN_TOOL, ErrorCode.INVALID_ARGUMENTS, ErrorCode.EPISODE_OVER,
}


@dataclass(frozen=True)
class Outcome:
    """Result of an agent write action."""

    ok: bool
    error: ErrorCode | None = None
    data: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def success(**data: Any) -> "Outcome":
        return Outcome(True, None, data)

    @staticmethod
    def fail(error: ErrorCode, **data: Any) -> "Outcome":
        return Outcome(False, error, data)


class World:
    """Mutable world state for one episode."""

    def __init__(
        self,
        *,
        location_names: dict[str, str],
        roads_open: dict[tuple[str, str], bool],
        station_type: dict[str, str],
        station_location: dict[str, str],
        station_online: dict[str, bool],
        item_type: dict[str, str],
        item_location: dict[str, str],
        agent_location: str,
        capacity: int,
        inventory: list[str] | None = None,
        stale_overrides: dict[str, str] | None = None,
    ) -> None:
        # Static topology
        self.location_names = location_names
        self._neighbors: dict[str, list[str]] = {loc: [] for loc in location_names}
        for a, b in roads_open:
            self._neighbors[a].append(b)
            self._neighbors[b].append(a)
        for loc in self._neighbors:
            self._neighbors[loc].sort()
        # Mutable state
        self.roads_open = roads_open
        self.station_type = station_type
        self.station_location = station_location
        self.station_online = station_online
        self.item_type = item_type
        self.item_location = item_location
        self.agent_location = agent_location
        self.capacity = capacity
        self.inventory: list[str] = list(inventory or [])  # pick order, oldest first
        self.stale_overrides: dict[str, str] = dict(stale_overrides or {})

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_scenario(cls, s: Scenario, *, validate: bool = True) -> "World":
        """Build the initial world. Validates the scenario first (raises ScenarioError), since
        e.g. duplicate item ids would otherwise silently collapse into one item."""
        if validate:
            check_scenario(s)
        m = s.map
        return cls(
            location_names={l.id: l.name for l in m.locations},
            roads_open={r.key: r.open for r in m.roads},
            station_type={st.id: st.type for st in m.stations},
            station_location={st.id: st.location for st in m.stations},
            station_online={st.id: st.online for st in m.stations},
            item_type={i.id: i.type for i in s.items},
            item_location={i.id: i.location for i in s.items},
            agent_location=s.start_location,
            capacity=s.capacity,
        )

    def clone(self) -> "World":
        """Independent copy (for the solver and for what-if checks by the injector)."""
        return World(
            location_names=self.location_names,  # never mutated, safe to share
            roads_open=dict(self.roads_open),
            station_type=self.station_type,  # never mutated
            station_location=self.station_location,  # never mutated
            station_online=dict(self.station_online),
            item_type=dict(self.item_type),
            item_location=dict(self.item_location),
            agent_location=self.agent_location,
            capacity=self.capacity,
            inventory=list(self.inventory),
            stale_overrides=dict(self.stale_overrides),
        )

    # ------------------------------------------------------------------ queries (harness)

    def neighbors(self, loc: str) -> list[str]:
        """All adjacent locations, open or closed, sorted."""
        return list(self._neighbors[loc])

    def open_neighbors(self, loc: str) -> list[str]:
        return [n for n in self._neighbors[loc] if self.roads_open[road_key(loc, n)]]

    def is_road_open(self, a: str, b: str) -> bool:
        return self.roads_open[road_key(a, b)]

    def items_at(self, loc: str) -> list[str]:
        """True item ids at a location, sorted."""
        return sorted(i for i, l in self.item_location.items() if l == loc)

    def stations_at(self, loc: str) -> list[str]:
        return sorted(s for s, l in self.station_location.items() if l == loc)

    def inventory_types(self) -> list[str]:
        return [self.item_type[i] for i in self.inventory]

    def map_item_location(self, item_id: str) -> str:
        """Where the map view says an item is (may differ from the truth)."""
        return self.stale_overrides.get(item_id, self.item_location[item_id])

    # ------------------------------------------------------------------ agent actions (W)

    def move(self, location: str) -> Outcome:
        here = self.agent_location
        if location not in self.location_names:
            return Outcome.fail(ErrorCode.UNKNOWN_ID, location=location)
        if location not in self._neighbors[here]:
            return Outcome.fail(ErrorCode.NOT_ADJACENT, location=location, current=here)
        if not self.roads_open[road_key(here, location)]:
            return Outcome.fail(ErrorCode.ROAD_CLOSED, location=location, current=here)
        self.agent_location = location
        return Outcome.success(
            location=location,
            previous=here,
            adjacent=[
                {"location": n, "open": self.roads_open[road_key(location, n)]}
                for n in self._neighbors[location]
            ],
        )

    def pick(self, item: str) -> Outcome:
        if item not in self.item_type:
            return Outcome.fail(ErrorCode.UNKNOWN_ID, item=item)
        if self.item_location[item] != self.agent_location:
            return Outcome.fail(ErrorCode.ITEM_NOT_HERE, item=item, location=self.agent_location)
        if len(self.inventory) >= self.capacity:
            return Outcome.fail(
                ErrorCode.INVENTORY_FULL, item=item, inventory=list(self.inventory),
                capacity=self.capacity,
            )
        self.item_location[item] = AGENT
        self.inventory.append(item)
        self.stale_overrides.pop(item, None)  # the agent now knows where this item is
        return Outcome.success(
            item=item, type=self.item_type[item], location=self.agent_location,
            inventory=list(self.inventory),
        )

    def drop(self, item: str) -> Outcome:
        if item not in self.item_type:
            return Outcome.fail(ErrorCode.UNKNOWN_ID, item=item)
        if item not in self.inventory:
            return Outcome.fail(ErrorCode.NOT_CARRYING, item=item)
        self.inventory.remove(item)
        self.item_location[item] = self.agent_location
        return Outcome.success(
            item=item, type=self.item_type[item], location=self.agent_location,
            inventory=list(self.inventory),
        )

    def use(self, station: str, items: list[str] | tuple[str, ...]) -> Outcome:
        items = list(items)
        if station not in self.station_type:
            return Outcome.fail(ErrorCode.UNKNOWN_ID, station=station)
        unknown = [i for i in items if i not in self.item_type]
        if unknown:
            return Outcome.fail(ErrorCode.UNKNOWN_ID, station=station, items=unknown)
        if self.station_location[station] != self.agent_location:
            return Outcome.fail(
                ErrorCode.NOT_AT_STATION, station=station, location=self.agent_location
            )
        if not self.station_online[station]:
            return Outcome.fail(ErrorCode.STATION_OFFLINE, station=station)
        missing = [i for i in items if i not in self.inventory]
        if missing:
            return Outcome.fail(ErrorCode.MISSING_ITEMS, station=station, items=missing)
        if len(set(items)) != len(items):
            return Outcome.fail(ErrorCode.INVALID_RECIPE, station=station, items=items)
        recipe = find_recipe(self.station_type[station], [self.item_type[i] for i in items])
        if recipe is None:
            return Outcome.fail(ErrorCode.INVALID_RECIPE, station=station, items=items)
        for i in items:
            self.inventory.remove(i)
            self.item_location[i] = CONSUMED
        output = self._next_item_id(recipe.output)
        self.item_type[output] = recipe.output
        self.item_location[output] = AGENT
        self.inventory.append(output)
        return Outcome.success(
            station=station, consumed=items, output=output, output_type=recipe.output,
            inventory=list(self.inventory),
        )

    def _next_item_id(self, item_type: str) -> str:
        n = 1
        while f"{item_type}_{n}" in self.item_type:
            n += 1
        return f"{item_type}_{n}"

    # ------------------------------------------------------------------ observers (R)

    def look(self) -> dict[str, Any]:
        """True state at the agent's location."""
        here = self.agent_location
        return {
            "location": here,
            "name": self.location_names[here],
            "items": [{"id": i, "type": self.item_type[i]} for i in self.items_at(here)],
            "stations": [
                {"id": s, "type": self.station_type[s], "online": self.station_online[s]}
                for s in self.stations_at(here)
            ],
            "roads": [
                {"to": n, "open": self.roads_open[road_key(here, n)]}
                for n in self._neighbors[here]
            ],
        }

    def status(self) -> dict[str, Any]:
        """The agent's own location and inventory. Job progress is added by the episode layer."""
        return {
            "location": self.agent_location,
            "inventory": [{"id": i, "type": self.item_type[i]} for i in self.inventory],
            "capacity": self.capacity,
        }

    def read_map(self) -> dict[str, Any]:
        """The map view: true state except stale item entries. Carried/consumed items omitted."""
        items = []
        for i in sorted(self.item_type):
            loc = self.map_item_location(i)
            if loc in (AGENT, CONSUMED):
                continue
            items.append({"id": i, "type": self.item_type[i], "location": loc})
        return {
            "locations": [
                {"id": l, "name": self.location_names[l]} for l in sorted(self.location_names)
            ],
            "roads": [
                {"a": a, "b": b, "open": self.roads_open[(a, b)]}
                for a, b in sorted(self.roads_open)
            ],
            "stations": [
                {
                    "id": s,
                    "type": self.station_type[s],
                    "location": self.station_location[s],
                    "online": self.station_online[s],
                }
                for s in sorted(self.station_type)
            ],
            "items": items,
        }

    # ------------------------------------------------------------------ harness-only mutators

    def close_road(self, a: str, b: str) -> None:
        self.roads_open[self._road(a, b)] = False

    def open_road(self, a: str, b: str) -> None:
        self.roads_open[self._road(a, b)] = True

    def set_station_online(self, station: str, online: bool) -> None:
        if station not in self.station_online:
            raise KeyError(f"unknown station {station!r}")
        self.station_online[station] = online

    def move_item(self, item: str, to: str, *, update_map: bool = True) -> None:
        """Relocate an item lying at a location.

        ``update_map=False`` is the stale-map disruption: the map keeps listing the item where it
        was. Repeated stale moves keep the *original* believed location.
        """
        if item not in self.item_location:
            raise KeyError(f"unknown item {item!r}")
        if to not in self.location_names:
            raise KeyError(f"unknown location {to!r}")
        old = self.item_location[item]
        if old in (AGENT, CONSUMED):
            raise ValueError(f"item {item!r} is {old}; only items at a location can be moved")
        if update_map:
            self.stale_overrides.pop(item, None)
        else:
            self.stale_overrides.setdefault(item, old)
        self.item_location[item] = to

    def _road(self, a: str, b: str) -> tuple[str, str]:
        key = road_key(a, b)
        if key not in self.roads_open:
            raise KeyError(f"no road between {a!r} and {b!r}")
        return key

    # ------------------------------------------------------------------ snapshot

    def snapshot(self) -> dict[str, Any]:
        """Complete, JSON-serializable, deterministically ordered state (for traces/evaluator)."""
        return {
            "agent_location": self.agent_location,
            "inventory": list(self.inventory),
            "capacity": self.capacity,
            "roads": [
                {"a": a, "b": b, "open": self.roads_open[(a, b)]} for a, b in sorted(self.roads_open)
            ],
            "stations": [
                {
                    "id": s,
                    "type": self.station_type[s],
                    "location": self.station_location[s],
                    "online": self.station_online[s],
                }
                for s in sorted(self.station_type)
            ],
            "items": [
                {"id": i, "type": self.item_type[i], "location": self.item_location[i]}
                for i in sorted(self.item_type)
            ],
            "stale_overrides": dict(sorted(self.stale_overrides.items())),
        }
