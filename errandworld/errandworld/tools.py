"""The agent-facing tool layer: tool definitions, argument validation, and text rendering.

This is the only place that turns world state into what an agent reads. Every strategy (ReAct,
adaptive replanning, local repair) sees exactly these definitions and these texts, so they must
stay neutral and deterministic.

- ``TOOL_SPECS``: the 7 tools (name, description, JSON-schema parameters, read/write kind).
  The MCP server (Phase 2) and LiteLLM function definitions are both built from these.
- ``dispatch(world, name, args)``: validate arguments, run the tool, return a ``ToolResult``.
  It never raises on agent input: unknown tools and malformed arguments come back as
  ``unknown_tool`` / ``invalid_arguments`` results that leave the world unchanged.
- ``ToolResult``: ``ok``, ``error``, structured ``data`` (for the trace logger), the ``text`` the
  LLM reads, and the ``resources`` the call touched (for labelling disruption-triggered replans).

Error texts always start with ``ERROR <code>:`` so a rule-based checker can match them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .types import road_key
from .world import ErrorCode, Outcome, World


class ToolKind(StrEnum):
    READ = "read"  # observes only; limited by the token cap
    WRITE = "write"  # changes the world; counts toward the action cap


@dataclass(frozen=True)
class ToolSpec:
    name: str
    kind: ToolKind
    description: str
    parameters: dict[str, Any]  # JSON schema (type: object)

    @property
    def required(self) -> tuple[str, ...]:
        return tuple(self.parameters.get("required", ()))


def _schema(properties: dict[str, Any] | None = None) -> dict[str, Any]:
    properties = properties or {}
    return {
        "type": "object",
        "properties": properties,
        "required": sorted(properties),
        "additionalProperties": False,
    }


_STR = {"type": "string"}

TOOL_SPECS: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in (
        ToolSpec(
            "move",
            ToolKind.WRITE,
            "Travel along an open road to an adjacent location. Counts as one action.",
            _schema({"location": {**_STR, "description": "Id of an adjacent location."}}),
        ),
        ToolSpec(
            "pick",
            ToolKind.WRITE,
            "Pick up an item at your current location. Fails if your inventory is full. "
            "Counts as one action.",
            _schema({"item": {**_STR, "description": "Id of an item at your location."}}),
        ),
        ToolSpec(
            "drop",
            ToolKind.WRITE,
            "Put down a carried item at your current location. Dropping the right item at a "
            "job's destination completes a delivery. Counts as one action.",
            _schema({"item": {**_STR, "description": "Id of an item you are carrying."}}),
        ),
        ToolSpec(
            "use",
            ToolKind.WRITE,
            "Run a station at your current location on carried items. The items are consumed "
            "and the station's output is added to your inventory. Counts as one action.",
            _schema(
                {
                    "station": {**_STR, "description": "Id of a station at your location."},
                    "items": {
                        "type": "array",
                        "items": _STR,
                        "minItems": 1,
                        "description": "Ids of the carried items to use as inputs.",
                    },
                }
            ),
        ),
        ToolSpec(
            "look",
            ToolKind.READ,
            "Observe your current location: items, stations (online or offline), and roads "
            "(open or closed). Does not count as an action.",
            _schema(),
        ),
        ToolSpec(
            "status",
            ToolKind.READ,
            "Report your location, inventory, and which jobs are done or pending. Does not count "
            "as an action.",
            _schema(),
        ),
        ToolSpec(
            "read_map",
            ToolKind.READ,
            "Read the town map: locations, roads, stations, and where items are listed. The map "
            "may be out of date. Does not count as an action.",
            _schema(),
        ),
    )
}

WRITE_TOOLS = frozenset(n for n, s in TOOL_SPECS.items() if s.kind is ToolKind.WRITE)
READ_TOOLS = frozenset(n for n, s in TOOL_SPECS.items() if s.kind is ToolKind.READ)


def function_definitions() -> list[dict[str, Any]]:
    """Tool definitions in the OpenAI/LiteLLM function-calling format, in a fixed order."""
    return [
        {
            "type": "function",
            "function": {"name": s.name, "description": s.description, "parameters": s.parameters},
        }
        for s in TOOL_SPECS.values()
    ]


# --------------------------------------------------------------------------- result


@dataclass(frozen=True)
class ToolResult:
    tool: str
    args: dict[str, Any]
    ok: bool
    error: ErrorCode | None
    data: dict[str, Any]
    text: str
    is_write: bool
    resources: tuple[str, ...] = field(default=())  # e.g. "road:home|market", "item:bulb_1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "args": self.args,
            "ok": self.ok,
            "error": self.error.value if self.error else None,
            "is_write": self.is_write,
            "resources": list(self.resources),
            "data": self.data,
            "text": self.text,
        }


# --------------------------------------------------------------------------- dispatch


def dispatch(
    world: World,
    name: str,
    args: dict[str, Any] | None = None,
    *,
    jobs: list[dict[str, Any]] | None = None,
) -> ToolResult:
    """Run one tool call against ``world``.

    ``jobs`` is the job-progress view (list of ``{"id", "description", "done"}``) that the episode
    layer supplies for ``status``; World itself does not track jobs.
    """
    raw_args = args
    args = {} if args is None else args
    spec = TOOL_SPECS.get(name) if isinstance(name, str) else None
    if spec is None:
        return _invalid(
            str(name), raw_args, ErrorCode.UNKNOWN_TOOL,
            f"Unknown tool {name!r}. Available tools: {', '.join(TOOL_SPECS)}.", is_write=False,
        )
    problem = _validate_args(spec, args)
    if problem:
        return _invalid(
            name, raw_args, ErrorCode.INVALID_ARGUMENTS, problem,
            is_write=spec.kind is ToolKind.WRITE,
        )

    if name == "move":
        here = world.agent_location
        o = world.move(args["location"])
        resources = [f"location:{args['location']}"]
        if args["location"] in world.location_names and args["location"] in world.neighbors(here):
            a, b = road_key(here, args["location"])
            resources.insert(0, f"road:{a}|{b}")
        return _write_result(spec, args, o, _render_move(o, here), resources)
    if name == "pick":
        o = world.pick(args["item"])
        return _write_result(spec, args, o, _render_pick(world, o), [f"item:{args['item']}"])
    if name == "drop":
        o = world.drop(args["item"])
        return _write_result(spec, args, o, _render_drop(world, o), [f"item:{args['item']}"])
    if name == "use":
        o = world.use(args["station"], args["items"])
        resources = [f"station:{args['station']}"] + [f"item:{i}" for i in args["items"]]
        return _write_result(spec, args, o, _render_use(world, o), resources)
    if name == "look":
        data = world.look()
        return _read_result(spec, args, data, _render_look(data))
    if name == "status":
        data = world.status()
        if jobs is not None:
            data["jobs"] = [dict(j) for j in jobs]
        return _read_result(spec, args, data, _render_status(data))
    if name == "read_map":
        data = world.read_map()
        return _read_result(spec, args, data, _render_map(data))
    raise AssertionError(f"tool {name!r} has a spec but no handler")  # pragma: no cover


def _validate_args(spec: ToolSpec, args: Any) -> str | None:
    if not isinstance(args, dict):
        return f"Arguments for {spec.name} must be an object, got {type(args).__name__}."
    props = spec.parameters["properties"]
    extra = sorted(set(args) - set(props))
    if extra:
        allowed = ", ".join(props) or "none"
        return f"Unexpected argument(s) for {spec.name}: {', '.join(extra)}. Allowed: {allowed}."
    missing = [k for k in spec.required if k not in args]
    if missing:
        return f"Missing argument(s) for {spec.name}: {', '.join(missing)}."
    for key, schema in props.items():
        value = args[key]
        if schema["type"] == "string" and not isinstance(value, str):
            return f"Argument {key!r} for {spec.name} must be a string."
        if schema["type"] == "array":
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                return f"Argument {key!r} for {spec.name} must be a list of strings."
            if len(value) < schema.get("minItems", 0):
                return f"Argument {key!r} for {spec.name} must not be empty."
    return None


def _invalid(
    name: str, args: Any, code: ErrorCode, message: str, *, is_write: bool
) -> ToolResult:
    safe_args = args if isinstance(args, dict) else {"_raw": repr(args)}
    return ToolResult(
        tool=name, args=dict(safe_args), ok=False, error=code, data={"message": message},
        text=f"ERROR {code.value}: {message}", is_write=is_write,
    )


def _write_result(
    spec: ToolSpec, args: dict, o: Outcome, text: str, resources: list[str]
) -> ToolResult:
    return ToolResult(
        tool=spec.name, args=dict(args), ok=o.ok, error=o.error, data=dict(o.data), text=text,
        is_write=True, resources=tuple(resources),
    )


def _read_result(spec: ToolSpec, args: dict, data: dict, text: str) -> ToolResult:
    return ToolResult(
        tool=spec.name, args=dict(args), ok=True, error=None, data=data, text=text,
        is_write=False,
    )


# --------------------------------------------------------------------------- rendering


def _err(code: ErrorCode, message: str) -> str:
    return f"ERROR {code.value}: {message}"


def _fmt_items(world: World, ids: list[str]) -> str:
    return ", ".join(f"{i} ({world.item_type[i]})" for i in ids) if ids else "empty"


def _fmt_inventory(world: World) -> str:
    return f"Inventory ({len(world.inventory)}/{world.capacity}): {_fmt_items(world, world.inventory)}."


def _fmt_roads(roads: list[dict]) -> str:
    return ", ".join(f"{r['to']} ({'open' if r['open'] else 'closed'})" for r in roads) or "none"


def _render_move(o: Outcome, here: str) -> str:
    if o.ok:
        roads = [{"to": a["location"], "open": a["open"]} for a in o.data["adjacent"]]
        return f"You moved to {o.data['location']}. Roads from here: {_fmt_roads(roads)}."
    target = o.data.get("location")
    if o.error is ErrorCode.UNKNOWN_ID:
        return _err(o.error, f"There is no location called {target!r}.")
    if o.error is ErrorCode.NOT_ADJACENT:
        return _err(o.error, f"{target} is not adjacent to {here}. You are still at {here}.")
    return _err(o.error, f"The road from {here} to {target} is closed. You are still at {here}.")


def _render_pick(world: World, o: Outcome) -> str:
    item = o.data.get("item")
    if o.ok:
        return f"You picked up {item} ({o.data['type']}). {_fmt_inventory(world)}"
    if o.error is ErrorCode.UNKNOWN_ID:
        return _err(o.error, f"There is no item called {item!r}.")
    if o.error is ErrorCode.ITEM_NOT_HERE:
        return _err(o.error, f"{item} is not at {world.agent_location}.")
    return _err(
        o.error, f"Your inventory is full ({world.capacity}/{world.capacity}); drop something "
        f"first. {_fmt_inventory(world)}",
    )


def _render_drop(world: World, o: Outcome) -> str:
    item = o.data.get("item")
    if o.ok:
        return f"You dropped {item} ({o.data['type']}) at {o.data['location']}. {_fmt_inventory(world)}"
    if o.error is ErrorCode.UNKNOWN_ID:
        return _err(o.error, f"There is no item called {item!r}.")
    return _err(o.error, f"You are not carrying {item}. {_fmt_inventory(world)}")


def _render_use(world: World, o: Outcome) -> str:
    st = o.data.get("station")
    if o.ok:
        return (
            f"{st} consumed {', '.join(o.data['consumed'])} and produced {o.data['output']} "
            f"({o.data['output_type']}). {_fmt_inventory(world)}"
        )
    e = o.error
    if e is ErrorCode.UNKNOWN_ID:
        if "items" in o.data:
            return _err(e, f"There is no item called {', '.join(map(repr, o.data['items']))}.")
        return _err(e, f"There is no station called {st!r}.")
    if e is ErrorCode.NOT_AT_STATION:
        return _err(e, f"{st} is not at {world.agent_location}.")
    if e is ErrorCode.STATION_OFFLINE:
        return _err(e, f"{st} is offline.")
    if e is ErrorCode.MISSING_ITEMS:
        return _err(e, f"You are not carrying {', '.join(o.data['items'])}. {_fmt_inventory(world)}")
    return _err(e, f"{st} ({world.station_type[st]}) has no recipe for those items.")


def _render_look(d: dict) -> str:
    items = ", ".join(f"{i['id']} ({i['type']})" for i in d["items"]) or "none"
    stations = (
        ", ".join(
            f"{s['id']} ({s['type']}, {'online' if s['online'] else 'offline'})"
            for s in d["stations"]
        )
        or "none"
    )
    return "\n".join(
        [
            f"You are at {d['location']}.",
            f"Items here: {items}.",
            f"Stations here: {stations}.",
            f"Roads from here: {_fmt_roads(d['roads'])}.",
        ]
    )


def _render_status(d: dict) -> str:
    inv = ", ".join(f"{i['id']} ({i['type']})" for i in d["inventory"]) or "empty"
    lines = [
        f"You are at {d['location']}.",
        f"Inventory ({len(d['inventory'])}/{d['capacity']}): {inv}.",
    ]
    if "jobs" in d:
        lines.append("Jobs:")
        for j in d["jobs"]:
            lines.append(f"- {j['id']} [{'done' if j['done'] else 'pending'}]: {j['description']}")
    return "\n".join(lines)


def _render_map(d: dict) -> str:
    lines = ["Locations: " + ", ".join(l["id"] for l in d["locations"]) + ".", "Roads:"]
    lines += [f"- {r['a']} <-> {r['b']} ({'open' if r['open'] else 'closed'})" for r in d["roads"]]
    lines.append("Stations:")
    lines += [
        f"- {s['id']} ({s['type']}) at {s['location']}, {'online' if s['online'] else 'offline'}"
        for s in d["stations"]
    ]
    lines.append("Items:")
    lines += [f"- {i['id']} ({i['type']}) at {i['location']}" for i in d["items"]] or ["- none"]
    return "\n".join(lines)
