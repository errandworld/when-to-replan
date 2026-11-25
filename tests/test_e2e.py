"""End to end through the real MCP server (FP3 1.6, in miniature).

A scripted model stands in for the LLM and plays the solver's own plan, so these tests check the
plumbing, not the prompts: server process, MCP client, shadow episode, goal check, budget,
revision path, and the trace files.
"""
import json

import pytest

pytest.importorskip("mcp")

from app.orchestrator import RunSpec, run_episode
from app.plan import PlanStep
from app.repair import repair_min
from app.types import ToolCall
from errandworld.episode import Episode
from errandworld.scenario_io import load_scenario
from errandworld.solver import plan as solver_plan

from tests.fakes import FakeLLM
from tests.helpers import DEV


def episode_and_plan(name):
    ep = Episode(load_scenario(DEV / name))
    return ep, solver_plan(ep.world, ep.tracker)


def plan_json(calls):
    return json.dumps({"reasoning": "scripted", "steps": [
        {"tool": t, "args": a, "preconditions": []} for t, a in calls]})


def test_react_solves_an_undisrupted_scenario(tmp_path):
    _, calls = episode_and_plan("dev_L10_D0_000.json")
    replies = [("", [ToolCall(t, a, f"c{i}")]) for i, (t, a) in enumerate(calls)] + ["Done."]
    record = run_episode(
        RunSpec(str(DEV / "dev_L10_D0_000.json"), "react", "scripted", out_dir=str(tmp_path)),
        llm=FakeLLM(replies),
    )
    assert record["success"] is True, record["error"]
    assert record["stop_reason"] == "goal_reached"
    assert record["divergences"] == 0          # server and shadow agree on every call
    assert record["infrastructure_failure"] is False
    assert record["write_calls"] == len(calls)
    assert record["llm_calls"] == len(calls)   # one model call per step, and it stops at the goal
    trace = json.loads((tmp_path / "trace.json").read_text())
    assert set(trace) >= {"header", "m1", "m2", "m3", "m4", "m5", "final_state"}
    assert trace["header"]["system_prompt_sha"] and trace["header"]["caps"]["action_cap"]
    assert trace["m5"]["success"] is True and trace["m5"]["failure_category"] is None


def test_adaptive_replans_once_after_a_disruption(tmp_path):
    name = "dev_L10_D1_000.json"
    ep, first = episode_and_plan(name)
    for tool, args in first:           # play the first plan until the world pushes back
        if not ep.call(tool, args).ok:
            break
    second = solver_plan(ep.world, ep.tracker)
    llm = FakeLLM([plan_json(first), plan_json(second)])
    record = run_episode(
        RunSpec(str(DEV / name), "adaptive", "scripted", out_dir=str(tmp_path)), llm=llm
    )
    assert record["success"] is True and record["stop_reason"] == "goal_reached", record["error"]
    assert record["divergences"] == 0
    assert record["revisions"] == 1 and record["disruptions_fired"] == 1
    trace = json.loads((tmp_path / "trace.json").read_text())
    assert [p["kind"] for p in trace["m4"]["plans"]] == ["initial", "rewrite"]
    assert trace["m4"]["revisions"][0]["trigger"]["condition"] == "tool_failed"


def test_token_cap_ends_the_episode_and_is_recorded_as_a_failure(tmp_path):
    _, calls = episode_and_plan("dev_L10_D0_000.json")
    replies = [("", [ToolCall(t, a, f"c{i}")]) for i, (t, a) in enumerate(calls)]
    record = run_episode(
        RunSpec(str(DEV / "dev_L10_D0_000.json"), "react", "scripted", token_cap=30, out_dir=str(tmp_path)),
        llm=FakeLLM(replies),  # FakeLLM charges 15 tokens a call
    )
    assert record["success"] is False and record["stop_reason"] == "token_cap", record["error"]
    assert record["failure_category"] == "budget_exhaustion"
    assert record["infrastructure_failure"] is False


class DownLLM:
    temperature = 0.0

    def complete(self, messages, **kwargs):
        raise ConnectionError("provider is down")


def test_an_api_outage_is_flagged_as_infrastructure_not_an_agent_failure(tmp_path):
    record = run_episode(
        RunSpec(str(DEV / "dev_L10_D0_000.json"), "react", "scripted", out_dir=str(tmp_path)),
        llm=DownLLM(),
    )
    assert record["infrastructure_failure"] is True
    assert record["stop_reason"] == "infrastructure_error"
    assert "ConnectionError" in record["error"]


def test_repair_min_is_the_true_distance_over_open_roads():
    ep, _ = episode_and_plan("dev_L10_D0_000.json")
    here = ep.world.agent_location
    neighbour = ep.world.neighbors(here)[0]
    assert repair_min(ep.world, PlanStep("move", {"location": neighbour})) == 1
    assert repair_min(ep.world, PlanStep("move", {"location": here})) == 0
    assert repair_min(ep.world, PlanStep("look", {})) is None  # reads are not write actions


def shortest(world, source, target):
    """Independent breadth-first search, so the test does not just call the code under test."""
    from collections import deque
    from errandworld.types import road_key
    seen, queue = {source}, deque([(source, 0)])
    while queue:
        node, d = queue.popleft()
        if node == target:
            return d
        for nxt in world.neighbors(node):
            if nxt not in seen and world.roads_open[road_key(node, nxt)]:
                seen.add(nxt)
                queue.append((nxt, d + 1))
    return None


def test_repair_min_for_use_is_the_walk_to_the_nearest_online_station_of_that_type_plus_one():
    from runner.scripted import first_with
    path = first_with("dev", "station_offline")
    ep = Episode(load_scenario(path))
    for tool, args in solver_plan(ep.world, ep.tracker):
        if ep.call(tool, args).ok:
            continue
        assert tool == "use", tool          # the offline station is what broke the plan
        w = ep.world
        online = [s for s, k in w.station_type.items() if k == w.station_type[args["station"]] and w.station_online[s]]
        expected = 1 + min(shortest(w, w.agent_location, w.station_location[s]) for s in online)
        assert repair_min(w, PlanStep("use", args)) == expected
        # no cap when the inputs are not carried or the station is unknown
        assert repair_min(w, PlanStep("use", {**args, "items": ["not_carried_9"]})) is None
        assert repair_min(w, PlanStep("use", {"station": "ghost_1", "items": []})) is None
        return
    raise AssertionError("the plan never failed")
