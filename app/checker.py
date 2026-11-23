"""What the agent has seen, and the rule-based checker that decides when a plan has broken.

Both work from tool results alone, never from the hidden world state.

Payload fields read here, by tool. This is a proposal to the world side until A.1 fixes the
tool return shapes; only Beliefs.update and its helpers read payloads, so a change stays here.

    move      location: str, adjacent: [str]
    look      location: str, items: [str], stations: [{name, online}], roads: [{to, open}]
    pick      inventory: [str]
    drop      inventory: [str], completed_job: str | null
    use       output: str, inventory: [str]
    status    location: str, inventory: [str], jobs: [{id, done}]
    read_map  roads: [{from, to, open}], items: [{item, location}], stations: [{name, location, online}]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from app.plan import PlanStep, Precondition
from app.types import ToolResult

TriggerCondition = Literal["tool_failed", "precondition_contradicted"]

# (time, value): time is the 1-based count of the tool result that reported the value.
_Timed = tuple[int, Any]


@dataclass(frozen=True)
class Trigger:
    """Why the checker fired. `step_index` is the plan step affected; `call_index` is the
    0-based position, among the episode's tool results, of the result that revealed the
    problem, which lets analysis tell disruption-triggered revisions from self-triggered ones."""

    condition: TriggerCondition
    step_index: int
    call_index: int
    detail: str  # the error code, or the contradicted precondition


class Beliefs:
    """Facts reported by the episode's tool results, the latest report of each fact winning.

    A fact no result has reported is unknown, and an unknown fact contradicts nothing. Results
    must be passed to update in the order they arrived, each exactly once.
    """

    def __init__(self) -> None:
        self._time = 0
        self._location: _Timed | None = None
        self._inventory: _Timed | None = None  # frozenset of items, from a full listing
        self._listings: dict[str, _Timed] = {}  # location -> frozenset of items seen by look
        self._facts: dict[tuple[str, ...], _Timed] = {}  # single facts -> bool

    @property
    def location(self) -> str | None:
        return self._location[1] if self._location else None

    def holds(self, precondition: Precondition) -> bool | None:
        """True or False if the results so far settle it, None if they do not."""
        found = self.evidence(precondition)
        return None if found is None else found[1]

    def evidence(self, precondition: Precondition) -> tuple[int, bool] | None:
        """The latest report bearing on a precondition, as (time, holds), or None."""
        a = precondition.args
        kind = precondition.type
        if kind == "at":
            if self._location is None:
                return None
            return self._location[0], self._location[1] == a.get("location")
        if kind == "carrying":
            return self._latest(("carrying", a.get("item", "")), self._inventory, a.get("item"))
        if kind == "item_at":
            listing = self._listings.get(a.get("location", ""))
            return self._latest(("item_at", a.get("item", ""), a.get("location", "")), listing, a.get("item"))
        if kind == "road_open":
            return self._facts.get(_road(a.get("from", ""), a.get("to", "")))
        if kind == "station_online":
            return self._facts.get(("station_online", a.get("station", "")))
        return None

    def update(self, result: ToolResult) -> None:
        """Fold one tool result into the beliefs."""
        self._time += 1
        payload = result.payload or {}
        here = self.location
        handler = getattr(self, f"_after_{result.tool}", None)
        if handler is not None:
            handler(result, payload, here)
        if result.ok:
            if isinstance(payload.get("location"), str):
                self._location = (self._time, payload["location"])
            if isinstance(payload.get("inventory"), list):
                self._inventory = (self._time, frozenset(map(str, payload["inventory"])))

    # One handler per tool, for the facts beyond location and inventory.

    def _after_move(self, result: ToolResult, payload: dict[str, Any], here: str | None) -> None:
        target = result.args.get("location")
        if result.ok:
            arrived = payload.get("location", target)
            self._set_road(here, arrived, True)
            if isinstance(arrived, str):
                self._location = (self._time, arrived)
        elif result.error in ("road_closed", "not_adjacent"):
            self._set_road(here, target, False)

    def _after_look(self, result: ToolResult, payload: dict[str, Any], here: str | None) -> None:
        if not result.ok:
            return
        where = payload.get("location", here)
        if isinstance(where, str) and isinstance(payload.get("items"), list):
            self._listings[where] = (self._time, frozenset(map(str, payload["items"])))
        self._set_stations(payload)
        for road in _records(payload.get("roads")):
            self._set_road(where, road.get("to"), road.get("open"))

    def _after_pick(self, result: ToolResult, payload: dict[str, Any], here: str | None) -> None:
        item = result.args.get("item")
        if result.ok:
            self._set(("carrying", str(item)), True)
        if (result.ok or result.error == "item_not_here") and here is not None:
            self._set(("item_at", str(item), here), False)

    def _after_drop(self, result: ToolResult, payload: dict[str, Any], here: str | None) -> None:
        item = result.args.get("item")
        if result.ok:
            self._set(("carrying", str(item)), False)
            if here is not None:
                self._set(("item_at", str(item), here), True)
        elif result.error == "not_carrying":
            self._set(("carrying", str(item)), False)

    def _after_use(self, result: ToolResult, payload: dict[str, Any], here: str | None) -> None:
        station = result.args.get("station")
        if result.ok:
            self._set(("station_online", str(station)), True)
            for item in _as_list(result.args.get("items")):
                self._set(("carrying", item), False)
            if isinstance(payload.get("output"), str):
                self._set(("carrying", payload["output"]), True)
        elif result.error == "station_offline":
            self._set(("station_online", str(station)), False)

    def _after_read_map(self, result: ToolResult, payload: dict[str, Any], here: str | None) -> None:
        if not result.ok:
            return
        for road in _records(payload.get("roads")):
            self._set_road(road.get("from"), road.get("to"), road.get("open"))
        for entry in _records(payload.get("items")):
            if isinstance(entry.get("item"), str) and isinstance(entry.get("location"), str):
                self._set(("item_at", entry["item"], entry["location"]), True)
        self._set_stations(payload)

    def _set_stations(self, payload: dict[str, Any]) -> None:
        for station in _records(payload.get("stations")):
            if isinstance(station.get("name"), str):
                self._set(("station_online", station["name"]), station.get("online"))

    def _set(self, key: tuple[str, ...], value: Any) -> None:
        if isinstance(value, bool):
            self._facts[key] = (self._time, value)

    def _set_road(self, a: Any, b: Any, value: Any) -> None:
        if isinstance(a, str) and isinstance(b, str):
            self._set(_road(a, b), value)

    def _latest(
        self, key: tuple[str, ...], snapshot: _Timed | None, member: Any
    ) -> tuple[int, bool] | None:
        """A single fact or a full listing, whichever is newer; a listing wins a tie, since it
        is the server's own report rather than an inference from the call."""
        fact = self._facts.get(key)
        if snapshot is not None and (fact is None or snapshot[0] >= fact[0]):
            return snapshot[0], member in snapshot[1]
        return fact


def failure(result: ToolResult, step_index: int, call_index: int) -> Trigger | None:
    """Fires when a tool call returned ok = false."""
    if result.ok:
        return None
    return Trigger("tool_failed", step_index, call_index, result.error or "unknown_error")


def contradiction(step: PlanStep, beliefs: Beliefs, step_index: int) -> Trigger | None:
    """Fires on the first of the step's preconditions that the results so far contradict."""
    for precondition in step.preconditions:
        found = beliefs.evidence(precondition)
        if found is not None and found[1] is False:
            return Trigger("precondition_contradicted", step_index, found[0] - 1, str(precondition))
    return None


def _road(a: str, b: str) -> tuple[str, ...]:
    """Roads are two-way, so a road is keyed by its ends in sorted order."""
    return ("road_open", *sorted((a, b)))


def _records(value: Any) -> list[dict[str, Any]]:
    return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    return [value] if isinstance(value, str) and value else []
