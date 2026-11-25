"""ReAct: one model call per step, the whole history in context."""
from __future__ import annotations

import json
import time
from typing import Any

from app.budget import Budget
from app.types import EpisodeOutcome, LLMResult, ToolCall, ToolResult


def run_react(
    *,
    task: str,
    system_prompt: str,
    llm: Any,
    tools: Any,
    budget: Budget,
) -> EpisodeOutcome:
    """Run one episode and return what happened."""
    definitions = tools.definitions()
    known = {d["function"]["name"] for d in definitions}
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]
    outcome = EpisodeOutcome(stop_reason="agent_stopped")
    started = time.perf_counter()

    while True:
        limit = budget.exceeded()
        if limit:
            outcome.stop_reason = limit
            break

        try:
            reply = llm.complete(messages, tools=definitions)
        except ValueError as exc:
            outcome.stop_reason, outcome.error = "loop_error", str(exc)
            break
        except Exception as exc:
            outcome.stop_reason = "infrastructure_error"
            outcome.error = f"{type(exc).__name__}: {exc}"
            break

        outcome.llm_calls.append(reply)
        budget.charge_llm(reply)

        if not reply.tool_calls:
            outcome.stop_reason = "agent_stopped"
            break

        for position, call in enumerate(reply.tool_calls):
            if not call.id:
                call.id = f"call_{len(outcome.tool_calls) + position}"
        messages.append(_assistant_message(reply))

        unknown = next((c for c in reply.tool_calls if c.name not in known), None)
        if unknown is not None:
            outcome.stop_reason, outcome.error = "loop_error", f"unknown tool: {unknown.name}"
            break

        for call in reply.tool_calls:
            result = tools.call(call.name, call.args)
            outcome.tool_calls.append(result)
            budget.charge_tool(result)
            messages.append(_tool_message(call, result))
            if budget.exceeded():
                break

    outcome.wall_clock_s = time.perf_counter() - started

    if outcome.llm_calls:
        outcome.final_output = outcome.llm_calls[-1].text
    return outcome


def _assistant_message(reply: LLMResult) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": reply.text}
    if reply.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.args)},
            }
            for call in reply.tool_calls
        ]
    return message


def _tool_message(call: ToolCall, result: ToolResult) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call.id, "content": observation_text(result)}


def observation_text(result: ToolResult) -> str:
    """What the model reads for one tool result: the server's text when it sent one, otherwise
    the ok flag, error code and payload as JSON (stand-in servers that have no text)."""
    if result.text:
        return result.text
    observation: dict[str, Any] = {"ok": result.ok}
    
    if result.error:
        observation["error"] = result.error
    observation.update({k: v for k, v in result.payload.items() if k not in observation})
    return json.dumps(observation)