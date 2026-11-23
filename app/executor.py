"""The executor: carries out plan steps as tool calls, one at a time, without calling the model."""
from __future__ import annotations

from typing import Any, Sequence

from app.budget import Budget
from app.checker import Beliefs
from app.plan import PlanStep
from app.types import ToolClient, ToolResult

# The read calls made before the planner is consulted: before the first plan, and before
# each revision. Read calls do not count against the action cap.
INITIAL_OBSERVATION = ("read_map", "status")
REVISION_OBSERVATION = ("status", "look")


class Executor:
    """Every call it makes is charged to the budget, kept in order in `results`, and folded
    into `beliefs`. It does not check the budget or the plan; the loop does. Whatever the tool
    client raises propagates, for the loop to record as an infrastructure failure."""

    def __init__(self, tools: ToolClient, budget: Budget):
        self.tools = tools
        self.budget = budget
        self.beliefs = Beliefs()
        self.results: list[ToolResult] = []

    def run(self, step: PlanStep) -> ToolResult:
        """Issue one plan step."""
        return self.call(step.tool, step.args)

    def observe(self, tool_names: Sequence[str]) -> list[ToolResult]:
        """Issue a fixed sequence of argument-free read calls."""
        return [self.call(name, {}) for name in tool_names]

    def call(self, name: str, args: dict[str, Any]) -> ToolResult:
        result = self.tools.call(name, dict(args))
        self.results.append(result)
        self.budget.charge_tool(result)
        self.beliefs.update(result)
        return result

    @property
    def last_index(self) -> int:
        """The position of the latest result in `results`, -1 before any call."""
        return len(self.results) - 1
