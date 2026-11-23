"""Plans: what the planner writes and the executor carries out.

A plan is a list of tool calls, each with the preconditions it expects to hold just before it
runs. Preconditions come from a small fixed vocabulary, so that a rule-based checker can test
them against what the tool results have shown.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# Each precondition type and the fields it takes. Every field is a string.
PRECONDITION_FIELDS: dict[str, tuple[str, ...]] = {
    "at": ("location",),
    "carrying": ("item",),
    "item_at": ("item", "location"),
    "road_open": ("from", "to"),
    "station_online": ("station",),
}


class PlanError(ValueError):
    """A reply that is not a valid plan. The message says what is wrong in terms the model can
    act on, so it can be sent back as feedback."""


@dataclass
class Precondition:
    type: str
    args: dict[str, str] = field(default_factory=dict)

    def __str__(self) -> str:
        names = PRECONDITION_FIELDS.get(self.type, tuple(self.args))
        return f"{self.type}({', '.join(self.args.get(n, '?') for n in names)})"

    def to_dict(self) -> dict[str, str]:
        return {"type": self.type, **self.args}


@dataclass
class PlanStep:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    preconditions: list[Precondition] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "args": dict(self.args),
            "preconditions": [p.to_dict() for p in self.preconditions],
        }


@dataclass
class Plan:
    """A parsed plan. Warnings list what parsing dropped without rejecting the plan."""

    steps: list[PlanStep] = field(default_factory=list)
    reasoning: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"reasoning": self.reasoning, "steps": [s.to_dict() for s in self.steps]}


def plan_schema(tool_definitions: list[dict[str, Any]]) -> dict[str, Any]:
    """The JSON schema of a plan reply, built from the tool definitions the server lists.

    A step's args take the union of every tool's parameters, because some providers reject
    per-tool variants; parse_plan then checks each step against its own tool. Preconditions
    are one variant per type, each requiring its own fields: with a single shared object,
    constrained decoding let models write a field from another type (item_at with "to").
    """
    arg_properties: dict[str, Any] = {}
    for definition in tool_definitions:
        arg_properties.update(_parameters(definition).get("properties", {}))

    precondition_variants = [
        {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": [kind]},
                **{name: {"type": "string"} for name in names},
            },
            "required": ["type", *names],
        }
        for kind, names in PRECONDITION_FIELDS.items()
    ]

    step = {
        "type": "object",
        "properties": {
            "tool": {"type": "string", "enum": [_name(d) for d in tool_definitions]},
            "args": {"type": "object", "properties": arg_properties},
            "preconditions": {"type": "array", "items": {"anyOf": precondition_variants}},
        },
        "required": ["tool", "args", "preconditions"],
    }
    return {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "steps": {"type": "array", "items": step},
        },
        "required": ["steps"],
    }


def parse_plan(text: str, tool_definitions: list[dict[str, Any]]) -> Plan:
    """Parse and validate a plan reply. Raises PlanError if it is not a usable plan.

    Arguments a step's tool does not take are dropped, and so is a precondition that states
    what the step itself achieves (move(X) expecting at(X), pick(I) expecting carrying(I));
    a warning records each non-empty drop. An empty step list is a valid plan.
    """
    data = _load_json(text)
    if not isinstance(data, dict) or not isinstance(data.get("steps"), list):
        raise PlanError('expected a JSON object with a "steps" list')

    tools = {_name(d): _parameters(d) for d in tool_definitions}
    plan = Plan(reasoning=str(data.get("reasoning") or ""))
    for index, raw in enumerate(data["steps"]):
        plan.steps.append(_parse_step(raw, f"step {index}", tools, plan.warnings))
    return plan


def _parse_step(
    raw: Any, where: str, tools: dict[str, dict[str, Any]], warnings: list[str]
) -> PlanStep:
    if not isinstance(raw, dict):
        raise PlanError(f"{where}: expected an object")
    tool = raw.get("tool")
    if tool not in tools:
        raise PlanError(f"{where}: unknown tool {tool!r}; expected one of {', '.join(tools)}")

    args = raw.get("args") or {}
    if not isinstance(args, dict):
        raise PlanError(f"{where}: args must be an object")
    allowed = tools[tool].get("properties", {})
    kept = {k: v for k, v in args.items() if k in allowed}
    dropped = sorted(k for k, v in args.items() if k not in allowed and not _empty(v))
    if dropped:
        warnings.append(f"{where}: dropped arguments {dropped}, which {tool} does not take")
    missing = [n for n in tools[tool].get("required", []) if _empty(kept.get(n))]
    if missing:
        raise PlanError(f"{where}: {tool} needs {', '.join(missing)}")

    raw_preconditions = raw.get("preconditions") or []
    if not isinstance(raw_preconditions, list):
        raise PlanError(f"{where}: preconditions must be a list")
    preconditions = []
    for raw_precondition in raw_preconditions:
        precondition = _parse_precondition(raw_precondition, where)
        if _is_own_effect(tool, kept, precondition):
            warnings.append(f"{where}: dropped {precondition}, which {tool} achieves rather than needs")
            continue
        preconditions.append(precondition)
    return PlanStep(tool=tool, args=kept, preconditions=preconditions)


def _parse_precondition(raw: Any, where: str) -> Precondition:
    if not isinstance(raw, dict):
        raise PlanError(f"{where}: each precondition must be an object")
    kind = raw.get("type")
    if kind not in PRECONDITION_FIELDS:
        raise PlanError(
            f"{where}: unknown precondition type {kind!r}; "
            f"expected one of {', '.join(PRECONDITION_FIELDS)}"
        )
    values = {}
    for name in PRECONDITION_FIELDS[kind]:
        value = raw.get(name)
        if not isinstance(value, str) or not value.strip():
            raise PlanError(f"{where}: precondition {kind} needs a {name}; got {json.dumps(raw)}")
        values[name] = value.strip()
    return Precondition(type=kind, args=values)


def _is_own_effect(tool: str, args: dict[str, Any], precondition: Precondition) -> bool:
    if tool == "move" and precondition.type == "at":
        return precondition.args["location"] == args.get("location")
    if tool == "pick" and precondition.type == "carrying":
        return precondition.args["item"] == args.get("item")
    return False


def _load_json(text: str) -> Any:
    """The JSON in a reply, tolerating a code fence or prose around a single object."""
    body = text.strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[1] if "\n" in body else ""
        body = body.rsplit("```", 1)[0]
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        start, end = body.find("{"), body.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(body[start : end + 1])
            except json.JSONDecodeError:
                pass
        raise PlanError(f"reply is not valid JSON: {exc.msg} at position {exc.pos}") from exc


def _name(definition: dict[str, Any]) -> str:
    return definition.get("function", {}).get("name", "")


def _parameters(definition: dict[str, Any]) -> dict[str, Any]:
    return definition.get("function", {}).get("parameters") or {}


def _empty(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}
