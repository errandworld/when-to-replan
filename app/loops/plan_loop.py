"""Adaptive replanning and local repair: one loop, differing only in the scope of a revision.

The planner writes a plan, the executor carries it out without calling the model, and before
and after each step the checker looks for a contradicted precondition or a failed call. When it
fires, the executor makes the fixed revision reads and the model is called once more:

    revision_scope="remaining"  adaptive replanning rewrites the whole rest of the plan
    revision_scope="step"       local repair replaces only the broken step, capped at the
                                solver's minimum plus two write actions; the rest stays frozen

Everything else, the planner, the checker, the prompts and the reads, is shared, so any
difference between the two conditions comes from the scope alone.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from app.budget import Budget
from app.checker import Trigger, contradiction, failure
from app.executor import INITIAL_OBSERVATION, REVISION_OBSERVATION, Executor
from app.plan import PlanError, PlanStep
from app.planner import Planner, ResponseFormat
from app.types import WRITE_TOOLS, EpisodeOutcome, PlanKind, PlanVersion, RevisionEvent, RevisionScope

# Given the broken step and the trigger, the fewest write actions that achieve the step's
# effect from the current world state. The harness computes it with the solver; the agent side
# cannot. None means no cap.
RepairLimit = Callable[[PlanStep, Trigger], "int | None"]

REPAIR_SLACK = 2  # FP2: a repair may take the solver's minimum plus two write actions


def run_plan_loop(
    *,
    task: str,
    system_prompt: str,
    llm: Any,
    tools: Any,
    budget: Budget,
    revision_scope: RevisionScope,
    repair_limit: RepairLimit | None = None,
    response_format: ResponseFormat = "schema",
) -> EpisodeOutcome:
    """Run one episode and return what happened.

    `system_prompt` is the plan_system prompt with its environment filled in; its {tools} slot
    is filled here from the server's definitions. `repair_limit` is used by local repair only.
    The episode ends when the plan runs out (agent_stopped), at a budget limit, on a plan that
    stays invalid after its retry (loop_error), or on a model or tool failure
    (infrastructure_error). The loop never stops on its own otherwise, so an uncapped run needs
    a step limit on the budget.
    """
    if revision_scope not in ("remaining", "step"):
        raise ValueError(f"unknown revision scope: {revision_scope!r}")

    outcome = EpisodeOutcome(stop_reason="agent_stopped")
    started = time.perf_counter()
    executor = Executor(tools, budget)
    episode = _Episode(
        outcome=outcome,
        executor=executor,
        budget=budget,
        scope=revision_scope,
        repair_limit=repair_limit,
        planner=Planner(
            llm=llm,
            system_prompt=system_prompt,
            task=task,
            tool_definitions=tools.definitions(),
            budget=budget,
            calls=outcome.llm_calls,
            response_format=response_format,
        ),
    )
    try:
        episode.run()
    except PlanError as exc:
        outcome.stop_reason = budget.exceeded() or "loop_error"
        outcome.error = str(exc)
    except Exception as exc:
        outcome.stop_reason = "infrastructure_error"
        outcome.error = f"{type(exc).__name__}: {exc}"

    outcome.tool_calls = list(executor.results)
    outcome.wall_clock_s = time.perf_counter() - started
    if outcome.llm_calls:
        outcome.final_output = outcome.llm_calls[-1].text
    return outcome


def run_adaptive(
    *,
    task: str,
    system_prompt: str,
    llm: Any,
    tools: Any,
    budget: Budget,
    response_format: ResponseFormat = "schema",
) -> EpisodeOutcome:
    """Adaptive replanning: on a trigger, rewrite the whole rest of the plan."""
    return run_plan_loop(
        task=task,
        system_prompt=system_prompt,
        llm=llm,
        tools=tools,
        budget=budget,
        revision_scope="remaining",
        response_format=response_format,
    )


def run_local_repair(
    *,
    task: str,
    system_prompt: str,
    llm: Any,
    tools: Any,
    budget: Budget,
    repair_limit: RepairLimit | None = None,
    response_format: ResponseFormat = "schema",
) -> EpisodeOutcome:
    """Local repair: on a trigger, replace only the broken step; the rest of the plan is frozen."""
    return run_plan_loop(
        task=task,
        system_prompt=system_prompt,
        llm=llm,
        tools=tools,
        budget=budget,
        revision_scope="step",
        repair_limit=repair_limit,
        response_format=response_format,
    )


@dataclass
class _Episode:
    outcome: EpisodeOutcome
    executor: Executor
    budget: Budget
    planner: Planner
    scope: RevisionScope
    repair_limit: RepairLimit | None
    steps: list[PlanStep] = field(default_factory=list)

    def run(self) -> None:
        self.executor.observe(INITIAL_OBSERVATION)
        if self._stopped():
            return
        calls_before = len(self.outcome.llm_calls)
        plan = self.planner.initial(self.executor.results)
        self.steps = list(plan.steps)
        self._record("initial", 0, plan.reasoning, plan.warnings, calls_before)

        cursor = 0
        while cursor < len(self.steps):
            if self._stopped():
                return
            step = self.steps[cursor]
            trigger = contradiction(step, self.executor.beliefs, cursor)
            if trigger is None:
                result = self.executor.run(step)
                trigger = failure(result, cursor, self.executor.last_index)
            if trigger is None:
                cursor += 1
            elif not self._stopped():  # a failed write may have used up the action cap
                self._revise(trigger, cursor)
        self.outcome.stop_reason = self.budget.exceeded() or "agent_stopped"

    def _revise(self, trigger: Trigger, cursor: int) -> None:
        """Replace steps from `cursor` on, as the scope allows; the cursor stays put."""
        self.executor.observe(REVISION_OBSERVATION)
        remaining = self.steps[cursor:]
        calls_before = len(self.outcome.llm_calls)
        cap, cut = None, 0
        if self.scope == "remaining":
            plan = self.planner.rewrite(
                history=self.executor.results, remaining=remaining, first=cursor, trigger=trigger
            )
            self.steps = self.steps[:cursor] + plan.steps
        else:
            minimum = self.repair_limit(remaining[0], trigger) if self.repair_limit else None
            cap = None if minimum is None else minimum + REPAIR_SLACK
            plan = self.planner.repair(
                history=self.executor.results,
                remaining=remaining,
                first=cursor,
                trigger=trigger,
                max_writes=cap,
            )
            patch = _within_cap(plan.steps, cap)
            cut = len(plan.steps) - len(patch)
            self.steps = self.steps[:cursor] + patch + self.steps[cursor + 1 :]
        kind = "rewrite" if self.scope == "remaining" else "repair"
        version = self._record(kind, cursor, plan.reasoning, plan.warnings, calls_before)
        self.outcome.revisions.append(RevisionEvent(trigger, self.scope, version, cap, cut))

    def _record(
        self, kind: PlanKind, start: int, reasoning: str, warnings: list[str], calls_before: int
    ) -> int:
        version = len(self.outcome.plans)
        self.outcome.plans.append(
            PlanVersion(
                version=version,
                kind=kind,
                start=start,
                steps=[s.to_dict() for s in self.steps],
                reasoning=reasoning,
                warnings=list(warnings),
                llm_calls=list(range(calls_before, len(self.outcome.llm_calls))),
            )
        )
        return version

    def _stopped(self) -> bool:
        limit = self.budget.exceeded()
        if limit:
            self.outcome.stop_reason = limit
        return limit is not None


def _within_cap(steps: list[PlanStep], cap: int | None) -> list[PlanStep]:
    """The longest prefix of `steps` holding at most `cap` write steps."""
    if cap is None:
        return list(steps)
    kept, writes = [], 0
    for step in steps:
        if step.tool in WRITE_TOOLS:
            if writes == cap:
                break
            writes += 1
        kept.append(step)
    return kept
