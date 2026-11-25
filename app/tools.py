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

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        start_timeout: float = 60.0,
        call_timeout: float = 120.0,
        errlog: Any = None,
    ):
        # A server that dies makes the client raise at once. These bound the other failure, a
        # server that is alive but stuck, so a hung episode fails (and is flagged as an
        # infrastructure failure) instead of freezing the whole experiment.
        self.errlog = errlog  # where the server's stderr goes; None means ours
        self.start_timeout = start_timeout
        self.call_timeout = call_timeout
        self.command = command
        self.args = list(args or [])
        self.env = env
        self._definitions: list[dict[str, Any]] = []
        self._session: Any = None
        self._stack: AsyncExitStack | None = None
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._started = threading.Event()
        self._start_error: BaseException | None = None
        self._stop: asyncio.Event | None = None
        self._lifecycle_future: Any = None

    def __enter__(self) -> "MCPToolClient":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def start(self) -> None:
        """Connect, in one long-lived task. anyio requires the task that opens a connection to
        be the one that closes it, so open and close both happen inside `_lifecycle`."""
        self._lifecycle_future = asyncio.run_coroutine_threadsafe(self._lifecycle(), self._loop)
        if not self._started.wait(self.start_timeout):
            self.close()
            raise TimeoutError(f"MCP server did not finish starting within {self.start_timeout}s")
        if self._start_error is not None:
            raise self._start_error

    async def _lifecycle(self) -> None:
        self._stop = asyncio.Event()
        try:
            await self._start()
        except BaseException as exc:  # reported to start(), which raises it
            self._start_error = exc
            self._started.set()
            if self._stack is not None:
                await self._stack.aclose()
                self._stack = None
            return
        self._started.set()
        await self._stop.wait()
        await self._stack.aclose()
        self._stack = None
        self._session = None

    def definitions(self) -> list[dict[str, Any]]:
        return list(self._definitions)

    def call(self, name: str, args: dict[str, Any]) -> ToolResult:
        start = time.perf_counter()
        raw = self._await(self._session.call_tool(name, args), self.call_timeout)
        body = _payload(raw)
        failed = bool(_attr(raw, "isError", "is_error"))
        if "ok" in body and "data" in body:  # the errandworld server's envelope (wire.py)
            return ToolResult(
                tool=name,
                args=dict(args),
                ok=bool(body["ok"]),
                error=body.get("error"),
                payload=body.get("data") or {},
                text=str(body.get("text", "")),
                latency_s=time.perf_counter() - start,
            )
        return ToolResult(  # any other server: whatever it sent becomes the payload
            tool=name,
            args=dict(args),
            ok=bool(body.get("ok", not failed)),
            error=body.get("error"),
            payload=body,
            latency_s=time.perf_counter() - start,
        )

    def close(self) -> None:
        if self._lifecycle_future is not None and self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
            try:
                self._lifecycle_future.result(timeout=20)
            except Exception:
                pass  # a server that will not exit cleanly is killed by the transport
            self._lifecycle_future = None
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)

    async def _start(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        self._stack = AsyncExitStack()
        params = StdioServerParameters(command=self.command, args=self.args, env=self.env)
        read, write = await self._stack.enter_async_context(
            stdio_client(params, **({"errlog": self.errlog} if self.errlog else {}))
        )
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        listed = await self._session.list_tools()
        self._definitions = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": _attr(tool, "inputSchema", "input_schema") or {"type": "object", "properties": {}},
                },
            }
            for tool in listed.tools
        ]

    def _await(self, coro: Any, timeout: "float | None" = None) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout)
        except TimeoutError:
            future.cancel()
            raise TimeoutError(f"MCP call did not finish within {timeout}s") from None


def _payload(raw: Any) -> dict[str, Any]:
    """The server's return as a dict, without interpreting it."""
    structured = _attr(raw, "structuredContent", "structured_content")
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


def _attr(obj: Any, *names: str) -> Any:
    """The first of several attribute names that exists. The mcp package renamed its fields
    from camelCase (1.x) to snake_case (2.x)."""
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None
