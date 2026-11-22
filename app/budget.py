"""Token, action and step limits for one episode."""
from __future__ import annotations

from dataclasses import dataclass

from app.types import LLMResult, StopReason, ToolResult


@dataclass
class Budget:
    """Counts an episode's usage, where a limit left as None is not enforced. Tokens are
    input plus output over every call, cached input included, and a write call counts whether
    or not it succeeded."""

    max_tokens: int | None = None
    max_write_actions: int | None = None
    max_steps: int | None = None

    tokens: int = 0
    write_actions: int = 0
    steps: int = 0

    def charge_llm(self, result: LLMResult) -> None:
        self.steps += 1
        self.tokens += result.input_tokens + result.cached_input_tokens + result.output_tokens

    def charge_tool(self, result: ToolResult) -> None:
        if result.is_write:
            self.write_actions += 1

    def exceeded(self) -> StopReason | None:
        if self.max_tokens is not None and self.tokens >= self.max_tokens:
            return "token_cap"
        if self.max_write_actions is not None and self.write_actions >= self.max_write_actions:
            return "action_cap"
        if self.max_steps is not None and self.steps >= self.max_steps:
            return "step_cap"
        return None
