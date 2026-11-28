"""Runs one episode: starts the server, sets the caps, runs a control loop, collects the traces.
 
The orchestrator is the only place that knows about both sides. It keeps the agent loops ignorant
of the simulator and the simulator ignorant of the agent.
 
Two Episode objects exist per run:
 
  * the real one, inside the MCP server process, that the agent plays against; and
  * a *shadow* in this process that is sent every call the agent makes.
 
The simulator is deterministic (same scenario + same actions => same observations and the same
disruption targets), so the shadow stays in lock-step with the server. That gives the harness what
it needs without a second channel into the server: the goal check (`is_success`), and the true
world for local repair's cap (`repair.py`). If the two ever disagree, that is a bug, and it is
counted in `divergences` instead of passing silently.
"""
from __future__ import annotations
 
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
 
from app.budget import Budget
from app.loops.plan_loop import run_adaptive, run_local_repair
from app.loops.react import run_react
from app.prompts import load
from app.repair import make_repair_limit
from app.tools import MCPToolClient
from app.trace import build_trace, sha, write_trace
from app.types import EpisodeOutcome, StopReason, ToolResult
from errandworld.episode import Episode
from errandworld.scenario_io import load_scenario
 
STRATEGIES = ("react", "adaptive", "local_repair")
UNCAPPED_ACTIONS = 10_000  # "uncapped" (FP3 2.5): no token cap, and an action cap nobody reaches
MAX_STEPS = 300            # safety net on LLM calls so a stuck loop cannot run forever
 
 
@dataclass(frozen=True)
class RunSpec:
    """Everything that identifies and configures one episode. (scenario, strategy, model, run)
    is the four-tuple that names it (FP3 3.2)."""
 
    scenario_path: str
    strategy: str
    model: str                 # key in configs/models.json
    run: int = 1
    token_cap: "int | None" = None
    uncapped: bool = False
    out_dir: str = "results/tmp"
    model_config: dict[str, Any] = field(default_factory=dict)
 
 
@dataclass
class GoalBudget(Budget):
    """Budget plus the harness's goal check: the diamond "goal reached or cap exceeded?" in the
    architecture diagram. The loops already ask `exceeded()` before every step, so success stops
    them with no change to the loops."""
 
    episode: Any = None
 
    def exceeded(self) -> "StopReason | None":
        if self.episode is not None and self.episode.is_success():
            return "goal_reached"
        return super().exceeded()
 
 
class MirroredTools:
    """The tool client the loops use: every call goes to the server and to the shadow."""
 
    def __init__(self, client: MCPToolClient, shadow: Episode):
        self.client, self.shadow = client, shadow
        self.divergences: list[dict[str, Any]] = []
 
    def definitions(self) -> list[dict[str, Any]]:
        return self.client.definitions()
 
    def call(self, name: str, args: dict[str, Any]) -> ToolResult:
        result = self.client.call(name, args)
        mirror = self.shadow.call(name, args)
        mirror_error = mirror.error.value if mirror.error else None
        if (result.ok, result.error, result.text) != (mirror.ok, mirror_error, mirror.text):
            self.divergences.append({"tool": name, "args": args, "server": result.text, "shadow": mirror.text})
        return result
 
 
def run_episode(spec: RunSpec, *, llm: Any, response_format: Any = "schema") -> dict[str, Any]:
    """Run one episode and return its run record (the JSONL line). The full trace is written to
    `<out_dir>/trace.json`. Nothing here raises for an infrastructure failure: it becomes a
    record flagged `infrastructure_failure` (FP3 4.7)."""
    if spec.strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {spec.strategy!r}; expected one of {STRATEGIES}")
    out_dir = Path(spec.out_dir)
    server_dir = out_dir / "server"
 
    scenario = load_scenario(spec.scenario_path)
    shadow = Episode(scenario, action_cap=UNCAPPED_ACTIONS if spec.uncapped else None)
    action_cap = shadow.action_cap
    token_cap = None if spec.uncapped else spec.token_cap
 
    environment = load("environment")
    system_prompt = load("react" if spec.strategy == "react" else "plan_system", environment=environment)
    task = shadow.task_description()
    budget = GoalBudget(
        episode=shadow, max_tokens=token_cap, max_write_actions=action_cap, max_steps=MAX_STEPS
    )
 
    outcome = EpisodeOutcome(stop_reason="infrastructure_error")
    tools: "MirroredTools | None" = None
    started = time.perf_counter()
    client = MCPToolClient(
        sys.executable,
        ["-m", "errandworld.server", "--scenario", str(spec.scenario_path),
         "--out", str(server_dir), "--action-cap", str(action_cap)],
        env=_server_env(),
    )
    try:
        client.start()
        tools = MirroredTools(client, shadow)
        common = dict(task=task, system_prompt=system_prompt, llm=llm, tools=tools, budget=budget)
        if spec.strategy == "react":
            outcome = run_react(**common)
        elif spec.strategy == "adaptive":
            outcome = run_adaptive(**common, response_format=response_format)
        else:
            outcome = run_local_repair(
                **common, repair_limit=make_repair_limit(shadow), response_format=response_format
            )
    except Exception as exc:  # server would not start, or died mid-episode
        outcome.stop_reason, outcome.error = "infrastructure_error", f"{type(exc).__name__}: {exc}"
    finally:
        client.close()
    wall = time.perf_counter() - started
 
    server_trace = _read_server_trace(server_dir)
    infra = outcome.stop_reason == "infrastructure_error" or server_trace is None
    divergences = tools.divergences if tools else []
    record: dict[str, Any] = {
        "scenario_id": scenario.id, "strategy": spec.strategy, "model": spec.model, "run": spec.run,
        "length_level": scenario.length_level, "disruption_level": scenario.disruption_level,
        "stop_reason": outcome.stop_reason, "infrastructure_failure": infra, "error": outcome.error,
        "wall_clock_s": round(wall, 3), "divergences": len(divergences),
        "caps": {"token_cap": token_cap, "action_cap": action_cap, "uncapped": spec.uncapped},
    }
    if server_trace is None:
        record.update(success=False, trace_path=None)
        return record
 
    success = bool(server_trace["summary"]["success"])
    if success != shadow.is_success():
        divergences.append({"tool": "<end>", "server": success, "shadow": shadow.is_success()})
        record["divergences"] = len(divergences)
 
    header = {
        "scenario_id": scenario.id, "strategy": spec.strategy, "model": spec.model, "run": spec.run,
        "scenario_path": str(spec.scenario_path),
        "scenario_fingerprint": server_trace["summary"]["scenario_fingerprint"],
        "model_config": spec.model_config,
        "model_versions_reported": sorted({c.model_version for c in outcome.llm_calls}),
        "temperature": getattr(llm, "temperature", None),
        "response_format": response_format if isinstance(response_format, str) else repr(response_format),
        "caps": record["caps"], "max_steps": MAX_STEPS,
        "system_prompt": system_prompt, "system_prompt_sha": sha(system_prompt),
        "environment_sha": sha(environment),
        "tool_definitions_sha": sha(json.dumps(tools.definitions(), sort_keys=True)) if tools else None,
        "divergences": divergences,
    }
    trace = build_trace(
        header=header, outcome=outcome, server_trace=server_trace, success=success,
        stop_reason=outcome.stop_reason,
    )
    trace_path = out_dir / "trace.json"
    write_trace(trace, trace_path)
 
    summary = server_trace["summary"]
    calls = outcome.llm_calls
    record.update(
        success=success,
        failure_category=trace["m5"]["failure_category"],
        write_calls=summary["write_calls"], read_calls=summary["read_calls"],
        invalid_write_calls=summary["invalid_write_calls"],
        progress_rate=summary["progress_rate"],
        disruptions_scheduled=summary["disruptions_scheduled"],
        disruptions_fired=summary["disruptions_fired"],
        llm_calls=len(calls),
        input_tokens=sum(c.input_tokens for c in calls),
        cached_input_tokens=sum(c.cached_input_tokens for c in calls),
        output_tokens=sum(c.output_tokens for c in calls),
        llm_seconds=round(sum(c.latency_s for c in calls), 3),
        tool_seconds=summary["tool_seconds"], injector_seconds=summary["injector_seconds"],
        revisions=len(outcome.revisions),
        # Local repair's hidden write cap: repairs it applied to, repairs it cut, steps it removed.
        repairs_capped=sum(r.write_cap is not None for r in outcome.revisions),
        repairs_truncated=sum(r.cut > 0 for r in outcome.revisions),
        steps_cut=sum(r.cut for r in outcome.revisions),
        recoverable_disruptions=sum(r["recovered"] is not None for r in trace["m5"]["recovery"]),
        recovered_disruptions=sum(r["recovered"] is True for r in trace["m5"]["recovery"]),
        final_output=outcome.final_output[:500],
        trace_path=str(trace_path),
    )
    return record
 
 
def _server_env() -> dict[str, str]:
    """The server's environment: ours, with the errandworld package on its path whether it was
    pip-installed or is just a folder in the repo."""
    import errandworld
 
    package_root = str(Path(errandworld.__file__).resolve().parent.parent)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (package_root, env.get("PYTHONPATH")) if p)
    return env
 
 
def _read_server_trace(server_dir: Path) -> "dict[str, Any] | None":
    path = server_dir / "trace.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None