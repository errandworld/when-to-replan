"""One episode: the boundary between an agent and ErrandWorld.

``Episode`` wires together the world, tool layer, progress tracker, and disruption injector for a
single scenario run.

Agent-facing (the only thing the MCP server will expose in Phase 2):
    ``call(tool, args) -> ToolResult``  and  ``task_description()``

Harness-facing (never shown to the agent):
    ``status`` / ``end_reason`` / ``is_success()``, ``end(reason)``, ``tool_calls`` (M2),
    ``disruption_events`` (M3), ``final_state()``, ``summary()``, ``trace()`` / ``save_trace()``

Rules
-----
- **Action cap** (default ``2 * reference_actions + 5``, where ``reference_actions`` is the
  optimal cost *with* the scheduled disruptions; FP2's ``2 x optimal + 5`` measured against the
  undisrupted optimum failed even the perfect oracle on short, heavily disrupted tasks) counts
  every write *attempt*: ``move``,
  ``pick``, ``drop``, ``use`` calls whether they succeed, fail, or have malformed arguments. Read
  tools never count (they are limited by the token cap, enforced by the agent harness). The call
  that uses the last allowed action still executes; if the task is not complete after it, the
  episode ends with ``end_reason="action_cap"``.
- **Disruptions** fire after a successful write whose subgoal progress reaches a trigger. The
  agent is not told; it meets the change on later calls. Injector time is recorded separately
  from tool time so disruptions do not inflate measured tool latency.
- **Ending**: success (all jobs done within the cap), ``action_cap``, or an external
  ``end(reason)`` from the harness (e.g. ``token_cap``, ``agent_stopped``, ``error``). After the
  episode ends, every call returns ``ERROR episode_over`` without touching the world, and
  unreached disruptions are recorded as ``not_reached``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .disruptions import DisruptionEvent, Injector
from .progress import ProgressTracker
from .scenario_io import fingerprint
from .solver import SolverConfig
from .tools import ToolResult, dispatch
from .types import Scenario
from .world import ErrorCode, World

RUNNING = "running"
SUCCESS = "success"
ACTION_CAP = "action_cap"


def default_action_cap(scenario: Scenario) -> int:
    """2 x reference cost + 5. The reference cost includes the scheduled disruptions; without
    disruptions it equals the optimum. Falls back to optimal_actions for hand-built scenarios."""
    ref = scenario.reference_actions if scenario.reference_actions is not None else (
        scenario.optimal_actions
    )
    if ref is None:
        raise ValueError(f"{scenario.id}: optimal/reference actions unknown; pass action_cap")
    return 2 * ref + 5


@dataclass
class ToolCallRecord:
    """One agent tool call (trace log M2)."""

    index: int  # 0-based position among all calls in the episode
    tool: str
    args: dict[str, Any]
    ok: bool
    error: str | None
    is_write: bool
    write_index: int | None  # 1-based count of write attempts after this call (writes only)
    resources: list[str]
    data: dict[str, Any]
    text: str
    subgoal_events: list[dict[str, Any]]
    subgoals_completed: int
    disruptions_fired: list[int]  # indices of disruptions that fired right after this call
    t_start: float  # seconds since episode start
    tool_seconds: float
    injector_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class Episode:
    scenario: Scenario
    action_cap: int | None = None
    solver_config: SolverConfig | None = None
    world: World = field(init=False)
    tracker: ProgressTracker = field(init=False)
    injector: Injector = field(init=False)
    tool_calls: list[ToolCallRecord] = field(init=False, default_factory=list)
    status: str = field(init=False, default=RUNNING)
    end_reason: str | None = field(init=False, default=None)
    write_count: int = field(init=False, default=0)
    read_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.action_cap is None:
            self.action_cap = default_action_cap(self.scenario)
        self.world = World.from_scenario(self.scenario)
        self.tracker = ProgressTracker(self.scenario.jobs)
        self.injector = Injector(self.scenario.schedule, self.scenario.seed, self.solver_config)
        self._t0 = time.perf_counter()

    # ------------------------------------------------------------------ agent-facing

    def task_description(self) -> str:
        """The task statement the agent receives at the start."""
        return "\n".join([
            f"You are at {self.scenario.start_location}. You can carry at most "
            f"{self.scenario.capacity} items.",
            self.tracker.task_description(),
        ])

    def call(self, tool: str, args: dict[str, Any] | None = None) -> ToolResult:
        if self.status != RUNNING:
            return ToolResult(
                tool=str(tool), args=dict(args) if isinstance(args, dict) else {}, ok=False,
                error=ErrorCode.EPISODE_OVER, data={"end_reason": self.end_reason},
                text=f"ERROR episode_over: The episode has ended ({self.end_reason}).",
                is_write=False,
            )
        t_start = time.perf_counter()
        result = dispatch(self.world, tool, args, jobs=self.tracker.jobs_view())
        sub_events = self.tracker.observe(result.tool, result.ok, result.data) if result.is_write else []
        tool_seconds = time.perf_counter() - t_start

        if result.is_write:
            self.write_count += 1
        else:
            self.read_count += 1

        fired: list[DisruptionEvent] = []
        t_inj = time.perf_counter()
        if result.is_write and result.ok and sub_events:
            fired = self.injector.fire_due(self.world, self.tracker, step=len(self.tool_calls))
        injector_seconds = time.perf_counter() - t_inj

        self.tool_calls.append(ToolCallRecord(
            index=len(self.tool_calls),
            tool=result.tool,
            args=result.args,
            ok=result.ok,
            error=result.error.value if result.error else None,
            is_write=result.is_write,
            write_index=self.write_count if result.is_write else None,
            resources=list(result.resources),
            data=result.data,
            text=result.text,
            subgoal_events=[e.to_dict() for e in sub_events],
            subgoals_completed=self.tracker.subgoals_completed(),
            disruptions_fired=[e.index for e in fired if e.fired],
            t_start=t_start - self._t0,
            tool_seconds=tool_seconds,
            injector_seconds=injector_seconds,
        ))

        if self.tracker.all_done():
            self._finish(SUCCESS, SUCCESS)
        elif self.write_count >= self.action_cap:
            self._finish(ACTION_CAP, ACTION_CAP)
        return result

    # ------------------------------------------------------------------ harness-facing

    def end(self, reason: str) -> None:
        """Stop a running episode from outside (token cap, agent gave up, harness error)."""
        if self.status == RUNNING:
            self._finish(reason, reason)

    def _finish(self, status: str, reason: str) -> None:
        self.status = status
        self.end_reason = reason
        self.injector.finalize(step=len(self.tool_calls))

    def is_success(self) -> bool:
        return self.status == SUCCESS

    @property
    def disruption_events(self) -> list[DisruptionEvent]:
        return self.injector.events

    def final_state(self) -> dict[str, Any]:
        return {"world": self.world.snapshot(), "progress": self.tracker.snapshot()}

    def summary(self) -> dict[str, Any]:
        writes = [c for c in self.tool_calls if c.is_write]
        events = self.disruption_events
        return {
            "scenario_id": self.scenario.id,
            "scenario_fingerprint": fingerprint(self.scenario),
            "length_level": self.scenario.length_level,
            "disruption_level": self.scenario.disruption_level,
            "status": self.status,
            "end_reason": self.end_reason,
            "success": self.is_success(),
            "optimal_actions": self.scenario.optimal_actions,
            "reference_actions": self.scenario.reference_actions,
            "action_cap": self.action_cap,
            "write_calls": self.write_count,
            "invalid_write_calls": sum(not c.ok for c in writes),
            "read_calls": self.read_count,
            "subgoals_completed": self.tracker.subgoals_completed(),
            "total_subgoals": self.tracker.total_subgoals(),
            "progress_rate": round(self.tracker.progress_rate(), 4),
            "jobs_done": [j.id for j in self.tracker.jobs if self.tracker.job_done(j.id)],
            "disruptions_scheduled": len(self.scenario.schedule),
            "disruptions_fired": sum(e.fired for e in events),
            "disruptions_fallback": sum(e.fired and e.fallback for e in events),
            "tool_seconds": round(sum(c.tool_seconds for c in self.tool_calls), 4),
            "injector_seconds": round(sum(c.injector_seconds for c in self.tool_calls), 4),
        }

    def trace(self) -> dict[str, Any]:
        """Everything the environment knows about the episode, JSON-serializable.

        The agent harness (Phase 2) adds LLM-call records (M1) and plan versions (M4)."""
        return {
            "summary": self.summary(),
            "task_description": self.task_description(),
            "tool_calls": [c.to_dict() for c in self.tool_calls],
            "disruptions": [e.to_dict() for e in self.disruption_events],
            "final_state": self.final_state(),
        }

    def save_trace(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.trace(), indent=2, default=str) + "\n", encoding="utf-8")
