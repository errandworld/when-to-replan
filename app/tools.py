"""The tool server, reached over MCP."""
from __future__ import annotations

import asyncio
import json
import threading
import time
from contextlib import AsyncExitStack
from typing import Any

from app.types import ToolResult


class MCPToolClient:
    """One server process per episode, started over stdio. Returns are passed through:
    whatever the server sends becomes the payload, and only the ok flag and the error code are
    read out of it."""

    def __init__(self, command: str, args: list[str] | None = None, env: dict[str, str] | None = None):
        self.command = command
        self.args = list(args or [])
        self.env = env
        self._definitions: list[dict[str, Any]] = []
        self._session: Any = None
        self._stack: AsyncExitStack | None = None
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    def __enter__(self) -> "MCPToolClient":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def start(self) -> None:
        self._await(self._start())

    def definitions(self) -> list[dict[str, Any]]:
        return list(self._definitions)

    def call(self, name: str, args: dict[str, Any]) -> ToolResult:
        start = time.perf_counter()
        raw = self._await(self._session.call_tool(name, args))
        payload = _payload(raw)
        return ToolResult(
            tool=name,
            args=dict(args),
            ok=bool(payload.get("ok", not getattr(raw, "isError", False))),
            error=payload.get("error"),
            payload=payload,
            latency_s=time.perf_counter() - start,
        )

    def close(self) -> None:
        if self._stack is not None:
            self._await(self._stack.aclose())
            self._stack = None
            self._session = None
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)

    async def _start(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        self._stack = AsyncExitStack()
        params = StdioServerParameters(command=self.command, args=self.args, env=self.env)
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        listed = await self._session.list_tools()
        self._definitions = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema or {"type": "object", "properties": {}},
                },
            }
            for tool in listed.tools
        ]

    def _await(self, coro: Any) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()


def _payload(raw: Any) -> dict[str, Any]:
    """The server's return as a dict, without interpreting it."""
    structured = getattr(raw, "structuredContent", None)
    if isinstance(structured, dict):
        return structured

    text = "".join(getattr(part, "text", "") for part in getattr(raw, "content", []) or [])
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"text": text}
    return parsed if isinstance(parsed, dict) else {"value": parsed}
