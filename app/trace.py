"""One trace file per episode: the five M records of FP2 section 4.2, plus the run-level fields.
 
    header   the four-tuple (scenario_id, strategy, model, run) and everything held fixed
    m1       every LLM call: cached and uncached input tokens, output tokens, latency
    m2       every tool call: arguments, the resource touched, whether it was valid   (server)
    m3       every disruption: scheduled and fired time, type, target                  (server)
    m4       every plan version, every replan or repair and the checker trigger        (loops)
    m5       final success, stop reason, failure category, recovery outcomes
 
M5 `recovery` has one entry per scheduled disruption: whether it fired, the jobs it affected,
and whether every affected job was done by the end (Table 3's recovery rate is the share of fired,
affecting disruptions that recovered). Recovery *cost* (write actions and tokens between a
disruption and the affected job's next subgoal, minus the solver's optimum for that stretch) is
not computed here: it needs the solver's optimum for each stretch. State that in 9.4.
"""
from __future__ import annotations
 
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any
 
from app.types import EpisodeOutcome
from metrics.failure import categorize
 
 
def build_trace(
    *,
    header: dict[str, Any],
    outcome: EpisodeOutcome,
    server_trace: dict[str, Any],
    success: bool,
    stop_reason: str,
) -> dict[str, Any]:
    trace: dict[str, Any] = {
        "header": header,
        "task_description": server_trace.get("task_description", ""),
        "m1": [
            {
                "call_index": i,
                "input_tokens": c.input_tokens,
                "cached_input_tokens": c.cached_input_tokens,
                "output_tokens": c.output_tokens,
                "latency_s": c.latency_s,
                "model_version": c.model_version,
            }
            for i, c in enumerate(outcome.llm_calls)
        ],
        "m2": server_trace["tool_calls"],
        "m3": server_trace["disruptions"],
        "m4": {
            "plans": [asdict(p) for p in outcome.plans],
            "revisions": [asdict(r) for r in outcome.revisions],
        },
        "m5": {
            "success": success,
            "stop_reason": stop_reason,
            "failure_category": None,
            "recovery": recovery_outcomes(server_trace),
        },
        "final_output": outcome.final_output,
        "final_state": server_trace["final_state"],
        "summary": server_trace["summary"],
    }
    trace["m5"]["failure_category"] = categorize(trace)
    return trace
 
 
def recovery_outcomes(server_trace: dict[str, Any]) -> list[dict[str, Any]]:
    """`recovered` is True or False when the disruption fired and affected at least one job, and
    None otherwise (it never fired, or it broke no job's every optimal plan)."""
    done = set(server_trace["summary"]["jobs_done"])
    outcomes = []
    for d in server_trace["disruptions"]:
        affected = list(d.get("affected_jobs") or [])
        outcomes.append({
            "disruption": d["index"], "type": d["type"], "fired": d["fired"], "target": d.get("target"),
            "affected_jobs": affected,
            "recovered": all(job in done for job in affected) if d["fired"] and affected else None,
        })
    return outcomes
 
 
def write_trace(trace: dict[str, Any], path: "str | Path") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trace, indent=1, default=str) + "\n", encoding="utf-8")
 
 
def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]