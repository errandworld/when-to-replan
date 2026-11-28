"""The planner: the model calls that write and revise plans.

Every request is stateless: the system prompt plus one user message built from the task and
the tool results so far. A reply that is not a valid plan gets one visible retry, with the
parse error sent back; the retry is charged to the budget like any other call.
"""
from __future__ import annotations

import json
from typing import Any, Literal, Optional, Sequence

from app.budget import Budget
from app.checker import Trigger
from app.plan import Plan, PlanError, PlanStep, parse_plan, plan_schema
from app.prompts import load
from app.types import LLMResult, ToolResult
from app.loops.react import observation_text

# How a plan is requested: "schema" sends the plan's JSON schema, "json" asks only for a JSON
# object, for models that cannot take a schema, and None sends no response format at all.
ResponseFormat = Optional[Literal["schema", "json"]]


class Planner:
    """Builds the three planning requests and turns the replies into plans.

    Every model call is charged to `budget` and appended to `calls`, the retry included. Plan
    requests raise PlanError if the reply is still invalid after the retry, or is invalid when
    the budget is already used up; anything the model client raises propagates.
    """

    def __init__(
        self,
        *,
        llm: Any,
        system_prompt: str,
        task: str,
        tool_definitions: list[dict[str, Any]],
        budget: Budget,
        calls: list[LLMResult],
        response_format: ResponseFormat = "schema",
    ):
        self.llm = llm
        self.task = task
        self.tool_definitions = tool_definitions
        self.budget = budget
        self.calls = calls
        self.system_prompt = system_prompt.replace("{tools}", render_tools(tool_definitions))
        self.response_format = _response_format(response_format, tool_definitions)

    def initial(self, observations: Sequence[ToolResult]) -> Plan:
        """A plan for the whole task, from the observations made so far."""
        return self._request(
            load("plan_initial", task=self.task, observations=render_results(observations))
        )

    def rewrite(
        self, *, history: Sequence[ToolResult], remaining: Sequence[PlanStep], first: int, trigger: Trigger
    ) -> Plan:
        """A new plan for everything left, to replace `remaining`, which starts at plan step
        `first`."""
        return self._request(
            load(
                "plan_rewrite",
                task=self.task,
                history=render_results(history),
                trigger=describe(trigger, remaining[0] if remaining else None),
                remaining=render_steps(remaining, first),
            )
        )

    def repair(
        self,
        *,
        history: Sequence[ToolResult],
        remaining: Sequence[PlanStep],
        first: int,
        trigger: Trigger,
    ) -> Plan:
        """Steps to replace only `remaining[0]`, the rest staying as they are.

        The request never states a write cap: the cap comes from hidden world state
        (`app/repair.py`), so the caller cuts a long repair afterwards instead."""
        return self._request(
            load(
                "plan_repair",
                task=self.task,
                history=render_results(history),
                trigger=describe(trigger, remaining[0] if remaining else None),
                failed=render_steps(remaining[:1], first),
                frozen=render_steps(remaining[1:], first + 1),
            )
        )

    def _request(self, content: str) -> Plan:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": content},
        ]
        for attempt in (1, 2):
            reply = self.llm.complete(messages, response_format=self.response_format)
            self.calls.append(reply)
            self.budget.charge_llm(reply)
            try:
                return parse_plan(reply.text, self.tool_definitions)
            except PlanError as exc:
                if attempt == 2 or self.budget.exceeded():
                    raise
                messages += [
                    {"role": "assistant", "content": reply.text},
                    {"role": "user", "content": load("plan_retry", error=str(exc))},
                ]
        raise AssertionError("unreachable")


def render_tools(tool_definitions: list[dict[str, Any]]) -> str:
    """The tool definitions as JSON, one tool per line."""
    return "\n".join(json.dumps(d.get("function", d)) for d in tool_definitions)


def render_results(results: Sequence[ToolResult]) -> str:
    """One entry per tool result, numbered by position, showing what ReAct's tool messages show."""
    from app.loops.react import observation_text    
    lines = []
    for index, result in enumerate(results):
        lines.append(
            f"[{index}] {result.tool} {json.dumps(result.args)} -> {observation_text(result)}"
        )
    return "\n".join(lines) or "(none)"


def render_steps(steps: Sequence[PlanStep], first: int) -> str:
    """One line per step, numbered by position in the plan."""
    return "\n".join(f"step {first + i}: {json.dumps(s.to_dict())}" for i, s in enumerate(steps)) or "(none)"


def describe(trigger: Trigger, step: PlanStep | None) -> str:
    """The trigger in words, naming the step and the tool result behind it."""
    call = f"{step.tool} {json.dumps(step.args)}" if step else "the next step"
    if trigger.condition == "tool_failed":
        return f"Step {trigger.step_index} ({call}) failed with error {trigger.detail} (result [{trigger.call_index}])."
    return (
        f"Step {trigger.step_index} ({call}) was not run: its precondition {trigger.detail} "
        f"is contradicted by result [{trigger.call_index}]."
    )


def _response_format(kind: ResponseFormat, tool_definitions: list[dict[str, Any]]) -> dict[str, Any] | None:
    if kind == "schema":
        return {"type": "json_schema", "json_schema": {"name": "plan", "schema": plan_schema(tool_definitions)}}
    if kind == "json":
        return {"type": "json_object"}
    if kind is None:
        return None
    raise ValueError(f"unknown response format: {kind!r}")
