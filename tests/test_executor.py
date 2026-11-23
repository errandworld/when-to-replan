import pytest

from app.budget import Budget
from app.executor import INITIAL_OBSERVATION, REVISION_OBSERVATION, Executor
from app.plan import PlanStep
from app.types import READ_TOOLS, ToolResult
from tests.fakes import FakeToolClient


def test_run_issues_the_step_and_records_it():
    move = ToolResult("move", {"location": "shop"}, ok=True, payload={"ok": True, "location": "shop"})
    tools = FakeToolClient({"move": move})
    budget = Budget(max_write_actions=1)
    executor = Executor(tools, budget)

    assert executor.last_index == -1
    result = executor.run(PlanStep("move", {"location": "shop"}))

    assert result is move
    assert [(c.name, c.args) for c in tools.calls] == [("move", {"location": "shop"})]
    assert executor.results == [move] and executor.last_index == 0
    assert budget.write_actions == 1 and budget.exceeded() == "action_cap"
    assert executor.beliefs.location == "shop"


def test_step_args_are_not_shared_with_the_tool_client():
    tools = FakeToolClient()
    step = PlanStep("pick", {"item": "bulb"})
    Executor(tools, Budget()).run(step)
    tools.calls[0].args["item"] = "changed"
    assert step.args == {"item": "bulb"}


def test_failed_write_is_charged_and_recorded():
    closed = ToolResult("move", {"location": "shop"}, ok=False, error="road_closed")
    budget = Budget()
    executor = Executor(FakeToolClient({"move": closed}), budget)
    executor.run(PlanStep("move", {"location": "shop"}))
    assert budget.write_actions == 1 and executor.results == [closed]


def test_observe_makes_free_read_calls():
    tools = FakeToolClient()
    budget = Budget(max_write_actions=1)
    executor = Executor(tools, budget)

    results = executor.observe(INITIAL_OBSERVATION)

    assert [(c.name, c.args) for c in tools.calls] == [("read_map", {}), ("status", {})]
    assert len(results) == 2 and executor.last_index == 1
    assert budget.write_actions == 0 and budget.exceeded() is None


def test_observation_sets_are_read_only():
    assert set(INITIAL_OBSERVATION) <= set(READ_TOOLS)
    assert set(REVISION_OBSERVATION) <= set(READ_TOOLS)


def test_tool_client_errors_propagate():
    class Broken(FakeToolClient):
        def call(self, name, args):
            raise ConnectionError("server exited")

    executor = Executor(Broken(), Budget())
    with pytest.raises(ConnectionError):
        executor.run(PlanStep("look"))
    assert executor.results == []
