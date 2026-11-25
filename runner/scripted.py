"""A model that is not a model: scripted replies built from the solver, for plumbing checks.
 
Used by the preflight and the end-to-end tests to drive all three control loops through the real
MCP server without calling an API. It checks that the pieces connect, not that any prompt works.
"""
from __future__ import annotations
 
import json
from collections import deque
from pathlib import Path
from typing import Any
 
from app.types import LLMResult, ToolCall
from errandworld.episode import Episode
from errandworld.scenario_io import load_scenario
from errandworld.solver import plan as solver_plan
from errandworld.types import road_key
 
SCENARIOS = Path(__file__).resolve().parents[1] / "errandworld" / "data" / "scenarios"
 
 
class ScriptedLLM:
    """Replies in order. A string is a plain reply, a list of (tool, args) is one tool call per
    entry, spread over successive replies as ReAct needs. Running out of replies raises."""
 
    temperature = 0.0
 
    def __init__(self, replies: list[Any]):
        self.replies = list(replies)
 
    def complete(self, messages: Any, *, tools: Any = None, response_format: Any = None) -> LLMResult:
        if not self.replies:
            raise AssertionError("ScriptedLLM ran out of replies")
        reply = self.replies.pop(0)
        text, calls = reply if isinstance(reply, tuple) else (reply, [])
        return LLMResult(text=text, tool_calls=list(calls), input_tokens=10, output_tokens=5,
                         model_version="scripted-0")
 
 
def scenario_path(set_name: str, name: str) -> Path:
    return SCENARIOS / set_name / name
 
 
def first_with(set_name: str, disruption_type: str, disruption_level: int = 1) -> Path:
    """The first scenario of a set whose first disruption has the given type."""
    for path in sorted((SCENARIOS / set_name).glob(f"*_D{disruption_level}_*.json")):
        if json.loads(path.read_text())["schedule"][0]["type"] == disruption_type:
            return path
    raise LookupError(f"no {set_name} scenario with a {disruption_type} disruption")
 
 
def plan_json(calls: list[tuple[str, dict[str, Any]]]) -> str:
    return json.dumps({"reasoning": "scripted", "steps": [
        {"tool": t, "args": a, "preconditions": []} for t, a in calls]})
 
 
def react_script(path: Path) -> list[Any]:
    """One tool call per reply, then a closing message: ReAct playing the solver's plan."""
    ep = Episode(load_scenario(path))
    calls = solver_plan(ep.world, ep.tracker)
    return [("", [ToolCall(t, a, f"c{i}")]) for i, (t, a) in enumerate(calls)] + ["Done."]
 
 
def adaptive_script(path: Path) -> list[Any]:
    """The solver's plan, then (after the first thing that fails) the solver's plan again from
    the world as it then stands. Meant for an adaptive run on a disrupted scenario."""
    ep = Episode(load_scenario(path))
    first = solver_plan(ep.world, ep.tracker)
    for tool, args in first:
        if not ep.call(tool, args).ok:
            break
    return [plan_json(first), plan_json(solver_plan(ep.world, ep.tracker))]
 
 
def repair_script(path: Path) -> "tuple[list[Any], int]":
    """For local repair on a road-closure scenario: the solver's plan, then a detour around the
    closed road for the one step that fails. Returns the replies and the detour's length, which
    is the solver minimum the harness should compute for that step."""
    ep = Episode(load_scenario(path))
    first = solver_plan(ep.world, ep.tracker)
    for tool, args in first:
        if ep.call(tool, args).ok:
            continue
        assert tool == "move", f"expected a closed road, but {tool} failed"
        detour = _detour(ep.world, ep.world.agent_location, args["location"])
        return [plan_json(first), plan_json([("move", {"location": n}) for n in detour])], len(detour)
    raise AssertionError("the first plan never failed, so there is nothing to repair")
 
 
def _detour(world: Any, source: str, target: str) -> list[str]:
    queue, parent = deque([source]), {source: None}
    while queue:
        node = queue.popleft()
        for nxt in world.neighbors(node):
            if nxt in parent or not world.roads_open[road_key(node, nxt)]:
                continue
            parent[nxt] = node
            if nxt == target:
                path = [nxt]
                while parent[path[-1]] != source:
                    path.append(parent[path[-1]])
                return path[::-1]
            queue.append(nxt)
    raise AssertionError(f"no open route from {source} to {target}")