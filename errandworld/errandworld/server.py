"""The ErrandWorld MCP server: one process per episode, over stdio.
 
    python -m errandworld.server --scenario <scenario.json> --out <dir> [--action-cap N]
 
The agent sees exactly the seven tools of ``tools.TOOL_SPECS``, with their schemas unchanged, and
nothing else. Setup, disruption injection and final-state reading are not tools: the Episode
lives inside this process, and the harness learns the hidden outcome only from ``<dir>/trace.json``,
which is rewritten after every call and once more when the client disconnects.
 
Requires the ``mcp`` package (``pip install -e ".[server]"``). Nothing may be printed to stdout:
the protocol runs over it.
"""
 
from __future__ import annotations
 
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
 
import anyio
from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
 
from .episode import Episode
from .scenario_io import load_scenario
from .tools import TOOL_SPECS
from .wire import to_wire
 
TRACE_FILE = "trace.json"
 
 
def write_trace(episode: Episode, out_dir: Path) -> None:
    """Atomically write the environment's record of the episode (M2, M3, final state)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / (TRACE_FILE + ".tmp")
    tmp.write_text(json.dumps(episode.trace(), indent=1, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, out_dir / TRACE_FILE)
 
 
def build_server(episode: Episode, out_dir: Path) -> Server:
    async def list_tools(ctx: Any, params: Any) -> types.ListToolsResult:
        return types.ListToolsResult(
            tools=[
                types.Tool(name=s.name, description=s.description, inputSchema=s.parameters)
                for s in TOOL_SPECS.values()
            ]
        )
 
    async def call_tool(ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        # Argument validation is errandworld's own (tools.dispatch), so a malformed call comes
        # back as an ordinary ``invalid_arguments`` result instead of a protocol error.
        result = episode.call(params.name, params.arguments)
        write_trace(episode, out_dir)
        wire = to_wire(result)
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(wire))],
            structuredContent=wire,
            isError=False,  # a failed tool call is news about the world, not a protocol error
        )
 
    return Server("errandworld", on_list_tools=list_tools, on_call_tool=call_tool)
 
 
async def serve(episode: Episode, out_dir: Path) -> None:
    server = build_server(episode, out_dir)
    write_trace(episode, out_dir)
    try:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())
    finally:
        # The client hung up: a token cap, a plan that ran out, or a crash. Close the books so
        # unreached disruptions are logged as not_reached.
        episode.end("terminated")
        write_trace(episode, out_dir)
 
 
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="ErrandWorld MCP server (one episode)")
    parser.add_argument("--scenario", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--action-cap", type=int, default=None)
    args = parser.parse_args(argv)
 
    episode = Episode(load_scenario(args.scenario), action_cap=args.action_cap)
    anyio.run(serve, episode, args.out)
 
 
if __name__ == "__main__":
    sys.exit(main())