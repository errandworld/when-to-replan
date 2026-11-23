"""Stand-ins for the model and the tool server."""
from __future__ import annotations
from typing import Any
from app.types import TOOLS, WRITE_TOOLS, LLMResult, ToolCall, ToolResult


class FakeLLM:
    """Scripted replies, returned in order. Each item is an LLMResult, a string, or a
    (text, tool_calls) pair, and running past the end of the script raises."""

    def __init__(self, replies: list[Any], model_version: str = "fake-model-0"):
        self.replies = list(replies)
        self.model_version = model_version
        self.calls: list[list[dict[str, Any]]] = []  # messages passed to each call

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        response_format: Any | None = None,
    ) -> LLMResult:
        self.calls.append(messages)
        if not self.replies:
            raise AssertionError(f"FakeLLM ran out of replies after {len(self.calls)} calls")
        reply = self.replies.pop(0)
        if isinstance(reply, LLMResult):
            return reply
        text, tool_calls = reply if isinstance(reply, tuple) else (reply, [])
        
        return LLMResult(
            text=text,
            tool_calls=list(tool_calls),
            input_tokens=10,
            output_tokens=5,
            model_version=self.model_version,
        )


class FakeToolClient:
    """Canned tool returns, and a record of the calls made. `returns` maps a tool name to a
    ToolResult, or to a list consumed one call at a time; anything else comes back ok with an
    empty payload."""

    def __init__(self, returns: dict[str, Any] | None = None):
        self.returns = dict(returns or {})
        self.calls: list[ToolCall] = []

    def definitions(self) -> list[dict[str, Any]]:
        """The seven tools with their argument names. Descriptions are placeholders."""
        args = {
            "move": ["location"],
            "look": [],
            "pick": ["item"],
            "drop": ["item"],
            "use": ["station", "items"],
            "status": [],
            "read_map": [],
        }
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"TODO: description for {name}",
                    "parameters": {
                        "type": "object",
                        "properties": {a: {"type": "string"} for a in args[name]},
                        "required": args[name],
                    },
                },
            }
            for name in TOOLS
        ]

    def call(self, name: str, args: dict[str, Any]) -> ToolResult:
        self.calls.append(ToolCall(name, dict(args)))
        canned = self.returns.get(name)
        if isinstance(canned, list):
            canned = canned.pop(0) if canned else None
        return canned if canned is not None else ToolResult(tool=name, args=dict(args), ok=True)

    @property
    def write_calls(self) -> list[ToolCall]:
        return [c for c in self.calls if c.name in WRITE_TOOLS]


class RecordingLLM(FakeLLM):
    """FakeLLM that also keeps the response format of each call, and copies the messages, so a
    caller that extends its message list afterwards does not change what was recorded."""

    def __init__(self, replies: list[Any], model_version: str = "fake-model-0"):
        super().__init__(replies, model_version)
        self.formats: list[Any] = []  # response_format passed to each call

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        response_format: Any | None = None,
    ) -> LLMResult:
        self.formats.append(response_format)
        return super().complete(list(messages), tools=tools, response_format=response_format)
