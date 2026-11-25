"""The wire format: what the MCP server sends back for every tool call.
 
Every call returns one JSON object, the same shape for all seven tools and for every error:
 
    {"ok": bool, "error": <code or null>, "text": str, "data": {...}}
 
- ``text``  is what the language model reads (``tools.ToolResult.text``). The agent loops show
            the model this and nothing else, so all three strategies see identical observations.
- ``data``  is the small set of facts the rule-based checker reads (``app/checker.py``,
            ``Beliefs.update``). Its field names are the checker's, not the simulator's, so the
            translation lives here, in one place, and neither side has to change when the other does.
            It is empty on failure; the error code carries the news.
 
Nothing here touches hidden state: every field is something the matching tool already revealed.
"""
 
from __future__ import annotations
 
from typing import Any
 
from .tools import ToolResult
 
 
def to_wire(result: ToolResult) -> dict[str, Any]:
    return {
        "ok": result.ok,
        "error": result.error.value if result.error else None,
        "text": result.text,
        "data": checker_view(result) if result.ok else {},
    }
 
 
def checker_view(result: ToolResult) -> dict[str, Any]:
    """The tool's data in the shape ``app.checker.Beliefs`` reads (see its module docstring)."""
    d, tool = result.data, result.tool
    if tool == "move":
        return {"location": d["location"], "adjacent": [a["location"] for a in d["adjacent"]]}
    if tool in ("pick", "drop"):
        return {"inventory": list(d["inventory"])}
    if tool == "use":
        return {"output": d["output"], "inventory": list(d["inventory"])}
    if tool == "look":
        return {
            "location": d["location"],
            "items": [i["id"] for i in d["items"]],
            "stations": [{"name": s["id"], "online": s["online"]} for s in d["stations"]],
            "roads": [{"to": r["to"], "open": r["open"]} for r in d["roads"]],
        }
    if tool == "status":
        return {
            "location": d["location"],
            "inventory": [i["id"] for i in d["inventory"]],
            "jobs": [{"id": j["id"], "done": j["done"]} for j in d.get("jobs", [])],
        }
    if tool == "read_map":
        return {
            "roads": [{"from": r["a"], "to": r["b"], "open": r["open"]} for r in d["roads"]],
            "items": [{"item": i["id"], "location": i["location"]} for i in d["items"]],
            "stations": [
                {"name": s["id"], "location": s["location"], "online": s["online"]}
                for s in d["stations"]
            ],
        }
    return {}