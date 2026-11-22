import json

from app.budget import Budget
from app.loops.react import run_react
from app.prompts import load
from app.types import ToolCall, ToolResult
from tests.fakes import FakeLLM, FakeToolClient

PROMPT = load("react", environment="[environment description]")


def react(replies, tools=None, budget=None):
    return run_react(
        task="Deliver the bulb to the library.",
        system_prompt=PROMPT,
        llm=FakeLLM(replies),
        tools=tools or FakeToolClient(),
        budget=budget or Budget(),
    )


def move(location="library"):
    return ToolCall("move", {"location": location})


def test_plain_reply_ends_episode():
    outcome = react(["Everything is delivered."])
    assert outcome.stop_reason == "agent_stopped"
    assert outcome.final_output == "Everything is delivered."
    assert len(outcome.llm_calls) == 1 and outcome.tool_calls == []


def test_tool_call_runs():
    tools = FakeToolClient()
    outcome = run_react(
        task="t",
        system_prompt=PROMPT,
        llm=FakeLLM([("Heading over.", [move()]), "Done."]),
        tools=tools,
        budget=Budget(),
    )
    assert [c.name for c in tools.calls] == ["move"]
    assert len(outcome.tool_calls) == 1 and outcome.tool_calls[0].ok
    assert outcome.stop_reason == "agent_stopped"


def test_observation_goes_back():
    tools = FakeToolClient({"move": ToolResult("move", {}, ok=False, error="road_closed")})
    llm = FakeLLM([("Heading over.", [move()]), "I will go around."])
    run_react(task="t", system_prompt=PROMPT, llm=llm, tools=tools, budget=Budget())

    last = llm.calls[-1][-1]
    assert last["role"] == "tool"
    assert json.loads(last["content"]) == {"ok": False, "error": "road_closed"}


def test_action_cap():
    outcome = react(
        [("One.", [move()]), ("Two.", [move()]), ("Three.", [move()])],
        budget=Budget(max_write_actions=2),
    )
    assert outcome.stop_reason == "action_cap"
    assert len(outcome.tool_calls) == 2


def test_token_cap():
    outcome = react([("One.", [move()]), ("Two.", [move()])], budget=Budget(max_tokens=15))
    assert outcome.stop_reason == "token_cap"
    assert len(outcome.llm_calls) == 1


def test_unknown_tool():
    outcome = react([("Trying.", [ToolCall("teleport", {})])])
    assert outcome.stop_reason == "loop_error" and "teleport" in outcome.error


def test_provider_failure():
    class Failing:
        def complete(self, messages, **kwargs):
            raise ConnectionError("gateway timeout")

    outcome = run_react(
        task="t", system_prompt=PROMPT, llm=Failing(), tools=FakeToolClient(), budget=Budget()
    )
    assert outcome.stop_reason == "infrastructure_error"
    assert "gateway timeout" in outcome.error
