import json

import pytest

from app.budget import Budget
from app.loops.plan_loop import run_adaptive, run_local_repair, run_plan_loop
from app.loops.react import run_react
from app.prompts import load
from app.types import EpisodeOutcome, ToolResult
from tests.fakes import FakeLLM, FakeToolClient

SYSTEM = load("plan_system", environment="[environment description]")


def step(tool, preconditions=(), **args):
    return {"tool": tool, "args": args, "preconditions": list(preconditions)}


def plan(*steps):
    return json.dumps({"reasoning": "r", "steps": list(steps)})


def closed(location):
    return ToolResult("move", {"location": location}, ok=False, error="road_closed",
                      payload={"ok": False, "error": "road_closed"})


def status(location):
    return ToolResult("status", {}, ok=True, payload={"ok": True, "location": location, "inventory": []})


def run(scope, replies, tools=None, budget=None, **kwargs):
    llm = FakeLLM(replies)
    tools = tools or FakeToolClient()
    outcome = run_plan_loop(
        task="Deliver the bulb to the office.",
        system_prompt=SYSTEM,
        llm=llm,
        tools=tools,
        budget=budget or Budget(),
        revision_scope=scope,
        **kwargs,
    )
    return outcome, llm, tools


def called(tools):
    return [(c.name, c.args.get("location") or c.args.get("item")) for c in tools.calls]


FETCH = plan(
    step("move", location="shop"),
    step("pick", item="bulb"),
    step("move", location="office"),
    step("drop", item="bulb"),
)


@pytest.mark.parametrize("scope", ["remaining", "step"])
def test_plan_runs_to_the_end(scope):
    outcome, _, tools = run(scope, [FETCH])

    assert outcome.stop_reason == "agent_stopped" and outcome.error is None
    assert called(tools) == [
        ("read_map", None), ("status", None),
        ("move", "shop"), ("pick", "bulb"), ("move", "office"), ("drop", "bulb"),
    ]
    assert len(outcome.tool_calls) == 6 and len(outcome.llm_calls) == 1
    assert outcome.revisions == []
    [initial] = outcome.plans
    assert initial.kind == "initial" and initial.version == 0 and initial.start == 0
    assert [s["tool"] for s in initial.steps] == ["move", "pick", "move", "drop"]
    assert initial.llm_calls == [0]
    assert outcome.final_output == FETCH


def test_adaptive_rewrites_the_rest_of_the_plan():
    tools = FakeToolClient({"status": [status("home"), status("home")], "move": [closed("shop")]})
    rewrite = plan(step("move", location="park"), step("move", location="shop"), step("pick", item="bulb"))
    outcome, llm, _ = run("remaining", [FETCH, rewrite], tools=tools)

    assert outcome.stop_reason == "agent_stopped"
    assert called(tools) == [
        ("read_map", None), ("status", None),
        ("move", "shop"),                       # fails
        ("status", None), ("look", None),       # revision reads
        ("move", "park"), ("move", "shop"), ("pick", "bulb"),
    ]
    [event] = outcome.revisions
    assert event.trigger.condition == "tool_failed" and event.trigger.detail == "road_closed"
    assert event.trigger.step_index == 0 and event.trigger.call_index == 2
    assert event.scope == "remaining" and event.plan_version == 1 and event.write_cap is None
    second = outcome.plans[1]
    assert second.kind == "rewrite" and second.start == 0 and second.llm_calls == [1]
    assert [s["args"].get("location") or s["args"].get("item") for s in second.steps] == ["park", "shop", "bulb"]
    assert "Write a new plan for everything that remains" in llm.calls[1][1]["content"]


def test_local_repair_replaces_only_the_broken_step():
    tools = FakeToolClient({"move": [closed("shop")]})
    patch = plan(step("move", location="park"), step("move", location="shop"))
    outcome, llm, _ = run("step", [FETCH, patch], tools=tools)

    assert outcome.stop_reason == "agent_stopped"
    assert called(tools)[2:] == [
        ("move", "shop"),
        ("status", None), ("look", None),
        ("move", "park"), ("move", "shop"),     # the patch
        ("pick", "bulb"), ("move", "office"), ("drop", "bulb"),  # the frozen rest
    ]
    [event] = outcome.revisions
    assert event.scope == "step" and event.cut == 0
    assert outcome.plans[1].kind == "repair"
    assert [s["tool"] for s in outcome.plans[1].steps] == ["move", "move", "pick", "move", "drop"]
    assert "Write steps that replace only the step to repair" in llm.calls[1][1]["content"]


def test_contradiction_fires_before_the_step_runs():
    tools = FakeToolClient({
        "status": [status("home")],
        "look": [ToolResult("look", {}, ok=True, payload={"ok": True, "roads": [{"to": "shop", "open": False}]})],
    })
    first = plan(
        step("look"),
        step("move", [{"type": "road_open", "from": "home", "to": "shop"}], location="shop"),
    )
    outcome, _, _ = run("remaining", [first, plan(step("move", location="park"))], tools=tools)

    assert ("move", "shop") not in called(tools)
    [event] = outcome.revisions
    assert event.trigger.condition == "precondition_contradicted"
    assert event.trigger.detail == "road_open(home, shop)"
    assert event.trigger.step_index == 1 and event.trigger.call_index == 2  # the look


def test_repair_is_capped_at_the_minimum_plus_two():
    tools = FakeToolClient({"move": [closed("shop")]})
    seen = []

    def repair_limit(broken, trigger):
        seen.append((broken.tool, broken.args, trigger.detail))
        return 1

    long_patch = plan(
        step("move", location="a"), step("look"), step("move", location="b"),
        step("move", location="c"), step("look"), step("move", location="d"), step("move", location="shop"),
    )
    outcome, llm, _ = run("step", [plan(step("move", location="shop")), long_patch],
                          tools=tools, repair_limit=repair_limit)

    assert seen == [("move", {"location": "shop"}, "road_closed")]
    assert "at most" not in llm.calls[1][1]["content"]  # cut afterwards, never stated
    [event] = outcome.revisions
    assert event.write_cap == 3 and event.cut == 2
    assert called(tools)[-5:] == [("move", "a"), ("look", None), ("move", "b"), ("move", "c"), ("look", None)]


def test_adaptive_ignores_the_repair_limit():
    tools = FakeToolClient({"move": [closed("shop")]})
    outcome, llm, _ = run("remaining", [plan(step("move", location="shop")), plan()],
                          tools=tools, repair_limit=lambda *_: 0)
    assert outcome.revisions[0].write_cap is None
    assert "at most" not in llm.calls[1][1]["content"]


def test_wrappers_pick_the_scope():
    common = dict(task="t", system_prompt=SYSTEM, budget=Budget())
    tools = FakeToolClient({"move": [closed("shop")]})
    adaptive = run_adaptive(llm=FakeLLM([plan(step("move", location="shop")), plan()]), tools=tools, **common)
    tools = FakeToolClient({"move": [closed("shop")]})
    repair = run_local_repair(llm=FakeLLM([plan(step("move", location="shop")), plan()]), tools=tools,
                              repair_limit=lambda *_: 1, **common)
    assert adaptive.revisions[0].scope == "remaining"
    assert repair.revisions[0].scope == "step" and repair.revisions[0].write_cap == 3


def test_empty_plan_stops_without_acting():
    outcome, _, tools = run("remaining", [plan()])
    assert outcome.stop_reason == "agent_stopped"
    assert called(tools) == [("read_map", None), ("status", None)]


def test_action_cap_stops_before_a_revision():
    tools = FakeToolClient({"move": [closed("shop")]})
    outcome, llm, _ = run("remaining", [FETCH], tools=tools, budget=Budget(max_write_actions=1))
    assert outcome.stop_reason == "action_cap"
    assert len(llm.calls) == 1 and outcome.revisions == []
    assert ("look", None) not in called(tools)


def test_action_cap_reached_on_the_last_step():
    outcome, _, _ = run("remaining", [plan(step("move", location="shop"))], budget=Budget(max_write_actions=1))
    assert outcome.stop_reason == "action_cap"


def test_token_cap_stops_before_acting():
    outcome, _, tools = run("remaining", [FETCH], budget=Budget(max_tokens=10))
    assert outcome.stop_reason == "token_cap"
    assert called(tools) == [("read_map", None), ("status", None)]


def test_invalid_plan_after_retry_is_a_loop_error():
    outcome, _, _ = run("remaining", ["not a plan", "still not a plan"])
    assert outcome.stop_reason == "loop_error" and "not valid JSON" in outcome.error
    assert len(outcome.llm_calls) == 2 and outcome.plans == []


def test_invalid_plan_with_the_budget_spent_is_a_cap():
    outcome, _, _ = run("remaining", ["not a plan"], budget=Budget(max_tokens=10))
    assert outcome.stop_reason == "token_cap" and outcome.error


def test_retry_is_recorded_on_the_plan_version():
    outcome, _, _ = run("remaining", ["not a plan", plan()])
    assert outcome.plans[0].llm_calls == [0, 1]


def test_tool_failure_is_an_infrastructure_error():
    class Broken(FakeToolClient):
        def call(self, name, args):
            if name == "pick":
                raise ConnectionError("server exited")
            return super().call(name, args)

    outcome, _, _ = run("remaining", [FETCH], tools=Broken())
    assert outcome.stop_reason == "infrastructure_error" and "server exited" in outcome.error
    assert [r.tool for r in outcome.tool_calls] == ["read_map", "status", "move"]


def test_model_failure_is_an_infrastructure_error():
    class Failing:
        def complete(self, messages, **kwargs):
            raise ConnectionError("gateway timeout")

    outcome = run_plan_loop(task="t", system_prompt=SYSTEM, llm=Failing(), tools=FakeToolClient(),
                            budget=Budget(), revision_scope="step")
    assert outcome.stop_reason == "infrastructure_error" and "gateway timeout" in outcome.error


def test_unknown_scope():
    with pytest.raises(ValueError):
        run("everything", [])


def test_every_loop_returns_the_same_outcome_type():
    outcome, _, _ = run("step", [plan()])
    assert type(outcome) is EpisodeOutcome

    react = run_react(task="t", system_prompt="s", llm=FakeLLM(["Done."]), tools=FakeToolClient(), budget=Budget())
    assert type(react) is EpisodeOutcome
    assert react.plans == [] and react.revisions == []  # ReAct leaves the plan records empty
