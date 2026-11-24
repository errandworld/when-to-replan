"""Episode: the agent/harness boundary, action cap, ending, logging, and determinism."""

import dataclasses
import glob
import json
from pathlib import Path

import pytest

from conftest import make_tiny_scenario
from errandworld.episode import ACTION_CAP, SUCCESS, Episode, default_action_cap
from errandworld.progress import ProgressTracker
from errandworld.scenario_io import load_scenario
from errandworld.solver import plan
from errandworld.types import DisruptionSpec, DisruptionType as DT
from errandworld.world import ErrorCode, World

ROOT = Path(__file__).resolve().parents[1]


def tiny(**kw):
    base = dict(optimal_actions=12, schedule=(), disruption_level=0)
    base.update(kw)
    return make_tiny_scenario(**base)


def optimal_calls(s):
    return plan(World.from_scenario(s), ProgressTracker(s.jobs))


def run_calls(ep, calls):
    return [ep.call(n, a) for n, a in calls]


# ---------------------------------------------------------------- basics


def test_default_action_cap():
    assert default_action_cap(tiny()) == 29
    with pytest.raises(ValueError):
        Episode(tiny(optimal_actions=None))
    assert Episode(tiny(optimal_actions=None), action_cap=7).action_cap == 7


def test_task_description():
    assert Episode(tiny()).task_description() == (
        "You are at home. You can carry at most 3 items.\n"
        "Complete all of these jobs:\n"
        "- job_1: Deliver a battery to office.\n"
        "- job_2: Make a lamp at a repair_bench from lamp_base + bulb, then deliver it to home."
    )


def test_optimal_play_succeeds():
    s = tiny()
    ep = Episode(s)
    results = run_calls(ep, optimal_calls(s))
    assert all(r.ok for r in results)
    assert ep.is_success() and ep.status == SUCCESS and ep.end_reason == SUCCESS
    sm = ep.summary()
    assert sm["write_calls"] == 12 and sm["read_calls"] == 0 and sm["invalid_write_calls"] == 0
    assert sm["progress_rate"] == 1.0 and sm["jobs_done"] == ["job_1", "job_2"]


def test_status_tool_shows_jobs():
    ep = Episode(tiny())
    r = ep.call("status", {})
    assert "- job_1 [pending]: Deliver a battery to office." in r.text


# ---------------------------------------------------------------- action cap


def test_reads_do_not_count_failed_writes_do():
    ep = Episode(tiny(), action_cap=3)
    for _ in range(10):
        ep.call("look", {})
        ep.call("read_map", {})
    assert ep.status == "running" and ep.write_count == 0 and ep.read_count == 20
    ep.call("move", {"location": "workshop"})  # not adjacent: failed write
    ep.call("move", {"bogus": 1})  # malformed write
    assert ep.write_count == 2 and ep.status == "running"
    ep.call("move", {"location": "market"})  # third write: cap reached, task unfinished
    assert ep.status == ACTION_CAP and not ep.is_success()
    assert ep.summary()["invalid_write_calls"] == 2


def test_success_on_the_last_allowed_action():
    s = tiny()
    ep = Episode(s, action_cap=12)
    run_calls(ep, optimal_calls(s))
    assert ep.is_success() and ep.write_count == 12


def test_one_action_short_fails():
    s = tiny()
    ep = Episode(s, action_cap=11)
    results = run_calls(ep, optimal_calls(s))
    assert ep.status == ACTION_CAP
    assert results[-1].error is ErrorCode.EPISODE_OVER  # 12th call rejected


def test_calls_after_end_are_rejected_and_harmless():
    s = tiny()
    ep = Episode(s)
    run_calls(ep, optimal_calls(s))
    before = ep.final_state()
    n_calls = len(ep.tool_calls)
    r = ep.call("move", {"location": "park"})
    assert not r.ok and r.error is ErrorCode.EPISODE_OVER
    assert r.text == "ERROR episode_over: The episode has ended (success)."
    assert ep.final_state() == before and len(ep.tool_calls) == n_calls


def test_external_end():
    ep = Episode(tiny())
    ep.call("look", {})
    ep.end("token_cap")
    assert ep.status == "token_cap" and ep.end_reason == "token_cap" and not ep.is_success()
    ep.end("other")  # no effect once ended
    assert ep.end_reason == "token_cap"
    assert ep.call("look", {}).error is ErrorCode.EPISODE_OVER


# ---------------------------------------------------------------- disruptions


def test_disruption_fires_and_is_logged():
    s = tiny(schedule=(DisruptionSpec(0, DT.ROAD_CLOSURE, 2),), disruption_level=1)
    ep = Episode(s)
    ep.call("move", {"location": "market"})
    ep.call("pick", {"item": "battery_1"})
    result = ep.call("pick", {"item": "lamp_base_1"})
    record = ep.tool_calls[-1]
    # The agent is not told: it sees exactly the normal pick result.
    assert result.text == (
        "You picked up lamp_base_1 (lamp_base). "
        "Inventory (2/3): battery_1 (battery), lamp_base_1 (lamp_base)."
    )
    events = ep.disruption_events
    assert len(events) == 1 and events[0].fired and events[0].step == 2
    assert record.disruptions_fired == [0] and record.subgoals_completed == 2
    assert record.injector_seconds >= 0 and record.tool_seconds >= 0


def test_unreached_disruptions_recorded_at_end():
    s = tiny(schedule=(DisruptionSpec(0, DT.ROAD_CLOSURE, 5),), disruption_level=1)
    ep = Episode(s, action_cap=2)
    ep.call("move", {"location": "market"})
    ep.call("move", {"location": "home"})
    assert ep.status == ACTION_CAP
    (ev,) = ep.disruption_events
    assert not ev.fired and ev.reason == "not_reached"
    assert ep.summary()["disruptions_fired"] == 0


def test_failed_writes_never_trigger_disruptions():
    s = tiny(schedule=(DisruptionSpec(0, DT.ROAD_CLOSURE, 1),), disruption_level=1)
    ep = Episode(s)
    ep.call("pick", {"item": "battery_1"})  # not here
    ep.call("drop", {"item": "battery_1"})
    assert ep.disruption_events == []


# ---------------------------------------------------------------- logging


def test_tool_call_records():
    ep = Episode(tiny())
    ep.call("look", {})
    ep.call("move", {"location": "market"})
    ep.call("pick", {"item": "battery_1"})
    look, move, pick = ep.tool_calls
    assert [c.index for c in ep.tool_calls] == [0, 1, 2]
    assert look.write_index is None and move.write_index == 1 and pick.write_index == 2
    assert move.resources == ["road:home|market", "location:market"]
    assert pick.subgoal_events[0]["kind"] == "pick" and pick.subgoals_completed == 1
    assert look.t_start <= move.t_start <= pick.t_start


def test_trace_is_json_and_complete(tmp_path):
    s = tiny(schedule=(DisruptionSpec(0, DT.MOVED_ITEM, 1),), disruption_level=1)
    ep = Episode(s)
    run_calls(ep, optimal_calls(s)[:3])
    ep.end("agent_stopped")
    path = tmp_path / "trace.json"
    ep.save_trace(path)
    tr = json.loads(path.read_text())
    assert set(tr) == {"summary", "task_description", "tool_calls", "disruptions", "final_state"}
    assert tr["summary"]["end_reason"] == "agent_stopped"
    assert len(tr["tool_calls"]) == 3 and len(tr["disruptions"]) == 1
    assert tr["final_state"]["progress"]["subgoals_completed"] >= 1


def test_traces_are_deterministic_apart_from_timing():
    s = tiny(schedule=(DisruptionSpec(0, DT.STALE_MAP, 1),), disruption_level=1)
    calls = optimal_calls(s)[:6] + [("look", {}), ("read_map", {})]

    def strip(trace):
        for c in trace["tool_calls"]:
            for k in ("t_start", "tool_seconds", "injector_seconds"):
                c.pop(k)
        for d in trace["disruptions"]:
            d.pop("solve_seconds")
        for k in ("tool_seconds", "injector_seconds"):
            trace["summary"].pop(k)
        return trace

    traces = []
    for _ in range(2):
        ep = Episode(s)
        run_calls(ep, calls)
        traces.append(strip(ep.trace()))
    assert traces[0] == traces[1]


# ---------------------------------------------------------------- real scenarios


@pytest.mark.parametrize("path", sorted(glob.glob(str(ROOT / "data/scenarios/pilot/pilot_L*_D3_*.json"))))
def test_replanning_oracle_succeeds_on_pilot_d3(path):
    """Follow an optimal plan and re-plan whenever a call fails or a disruption fires."""
    s = load_scenario(path)
    ep = Episode(s)
    calls = plan(ep.world, ep.tracker)
    while ep.status == "running":
        name, args = calls.pop(0)
        r = ep.call(name, args)
        rec = ep.tool_calls[-1]
        if not r.ok or rec.disruptions_fired:
            calls = plan(ep.world, ep.tracker)
    assert ep.is_success(), ep.summary()
    sm = ep.summary()
    assert sm["disruptions_fired"] == 3 and sm["disruptions_fallback"] == 0
    assert sm["write_calls"] <= ep.action_cap
