import json

import pytest

from app.budget import Budget
from app.checker import Trigger
from app.plan import PlanError, PlanStep, Precondition
from app.planner import Planner, describe, render_results, render_steps, render_tools
from app.prompts import load
from app.types import ToolResult
from tests.fakes import FakeToolClient, RecordingLLM

TOOLS = FakeToolClient().definitions()
SYSTEM = load("plan_system", environment="[environment description]")
TASK = "Deliver the bulb to the office."


def plan_text(*tools_and_args):
    return json.dumps(
        {"reasoning": "r", "steps": [{"tool": t, "args": a, "preconditions": []} for t, a in tools_and_args]}
    )


def planner(replies, budget=None, response_format="schema"):
    llm = RecordingLLM(replies)
    calls = []
    p = Planner(
        llm=llm,
        system_prompt=SYSTEM,
        task=TASK,
        tool_definitions=TOOLS,
        budget=budget or Budget(),
        calls=calls,
        response_format=response_format,
    )
    return p, llm, calls


OBSERVED = [
    ToolResult("read_map", {}, ok=True, payload={"ok": True, "roads": []}),
    ToolResult("status", {}, ok=True, payload={"ok": True, "location": "home"}),
]
FAILED = Trigger("tool_failed", step_index=2, call_index=4, detail="road_closed")
REMAINING = [PlanStep("move", {"location": "shop"}), PlanStep("pick", {"item": "bulb"})]


def test_initial_request():
    p, llm, calls = planner([plan_text(("move", {"location": "shop"}))])
    plan = p.initial(OBSERVED)

    assert [s.tool for s in plan.steps] == ["move"]
    system, user = llm.calls[0]
    assert system["role"] == "system" and user["role"] == "user"
    assert "{tools}" not in system["content"] and '"name": "read_map"' in system["content"]
    assert "[environment description]" in system["content"]
    assert TASK in user["content"]
    assert '[1] status {} -> {"ok": true, "location": "home"}' in user["content"]
    assert len(calls) == 1


def test_calls_are_charged():
    budget = Budget()
    p, _, calls = planner([plan_text()], budget=budget)
    p.initial([])
    assert budget.steps == 1 and budget.tokens == 15 and len(calls) == 1


@pytest.mark.parametrize(
    "kind, expected",
    [("json", {"type": "json_object"}), (None, None)],
)
def test_response_formats(kind, expected):
    p, llm, _ = planner([plan_text()], response_format=kind)
    p.initial([])
    assert llm.formats == [expected]


def test_schema_response_format():
    p, llm, _ = planner([plan_text()])
    p.initial([])
    sent = llm.formats[0]
    assert sent["type"] == "json_schema" and sent["json_schema"]["name"] == "plan"
    assert "steps" in sent["json_schema"]["schema"]["properties"]


def test_unknown_response_format():
    with pytest.raises(ValueError):
        planner([], response_format="xml")


def test_invalid_reply_gets_one_retry():
    p, llm, calls = planner(["not a plan", plan_text(("look", {}))])
    plan = p.initial([])

    assert [s.tool for s in plan.steps] == ["look"]
    assert len(calls) == 2
    retry = llm.calls[1]
    assert retry[2] == {"role": "assistant", "content": "not a plan"}
    assert retry[3]["role"] == "user" and "not valid JSON" in retry[3]["content"]


def test_invalid_twice_raises():
    p, _, calls = planner(["not a plan", '{"steps": [{"tool": "teleport"}]}'])
    with pytest.raises(PlanError, match="teleport"):
        p.initial([])
    assert len(calls) == 2


def test_no_retry_once_the_budget_is_spent():
    p, _, calls = planner(["not a plan", plan_text()], budget=Budget(max_tokens=10))
    with pytest.raises(PlanError):
        p.initial([])
    assert len(calls) == 1


def test_model_errors_propagate():
    class Failing:
        def complete(self, messages, **kwargs):
            raise ConnectionError("gateway timeout")

    p = Planner(llm=Failing(), system_prompt=SYSTEM, task=TASK, tool_definitions=TOOLS, budget=Budget(), calls=[])
    with pytest.raises(ConnectionError):
        p.initial([])


def test_rewrite_request():
    p, llm, _ = planner([plan_text(("move", {"location": "park"}))])
    plan = p.rewrite(history=OBSERVED, remaining=REMAINING, first=2, trigger=FAILED)

    assert [s.tool for s in plan.steps] == ["move"]
    content = llm.calls[0][1]["content"]
    assert "[0] read_map {}" in content
    assert 'Step 2 (move {"location": "shop"}) failed with error road_closed (result [4]).' in content
    assert 'step 2: {"tool": "move"' in content and 'step 3: {"tool": "pick"' in content
    assert "replaces the rest of the current plan" in content


def test_repair_request_splits_the_broken_step_from_the_frozen_rest_and_states_no_cap():
    p, llm, _ = planner([plan_text(("move", {"location": "park"}))])
    p.repair(history=OBSERVED, remaining=REMAINING, first=2, trigger=FAILED)

    content = llm.calls[0][1]["content"]
    failed = content.split("## The step to repair")[1].split("##")[0]
    frozen = content.split("will run unchanged after your repair")[1]
    assert 'step 2: {"tool": "move"' in failed and "pick" not in failed
    assert 'step 3: {"tool": "pick"' in frozen
    assert "at most" not in content and "{limit}" not in content  # the cap is hidden state


def test_repair_request_with_no_history_and_nothing_frozen():
    p, llm, _ = planner([plan_text()])
    p.repair(history=[], remaining=REMAINING[:1], first=0, trigger=FAILED)

    content = llm.calls[0][1]["content"]
    assert "at most" not in content
    assert "(none)" in content  # no history, nothing frozen


def test_render_results_shows_errors_and_payload():
    result = ToolResult("move", {"location": "shop"}, ok=False, error="road_closed", payload={"ok": False, "error": "road_closed"})
    assert render_results([result]) == '[0] move {"location": "shop"} -> {"ok": false, "error": "road_closed"}'
    assert render_results([]) == "(none)"


def test_render_steps_and_tools():
    step = PlanStep("pick", {"item": "bulb"}, [Precondition("at", {"location": "shop"})])
    assert render_steps([step], 5) == (
        'step 5: {"tool": "pick", "args": {"item": "bulb"}, '
        '"preconditions": [{"type": "at", "location": "shop"}]}'
    )
    assert render_steps([], 0) == "(none)"
    assert len(render_tools(TOOLS).splitlines()) == len(TOOLS)


def test_describe_contradiction():
    trigger = Trigger("precondition_contradicted", step_index=1, call_index=3, detail="road_open(home, shop)")
    assert describe(trigger, REMAINING[0]) == (
        'Step 1 (move {"location": "shop"}) was not run: its precondition road_open(home, shop) '
        "is contradicted by result [3]."
    )
