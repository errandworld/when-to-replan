"""Job progress: subgoals, latched job completion, and job descriptions.

Subgoals
--------
- DELIVER job (target T, destination D): ``pick T`` -> ``deliver T@D``          (2 subgoals)
- CRAFT job (output O from inputs I1, I2):  ``pick I1``, ``pick I2`` -> ``use O`` -> ``deliver O@D``
                                                                                (4 subgoals)

Progress is measured in subgoals completed. Disruption triggers (``disruptions.py``) and the
progress-rate metric both use this count, so it must be **monotone** (never decreases) and must
not be inflated by redundant actions.

Crediting rules
---------------
Items are interchangeable by type, and several jobs may need the same type, so the tracker
credits *by type* and resolves ambiguity deterministically:

1. A successful ``pick`` of type T credits the first pending job, in job order, that has an
   uncredited ``pick T`` subgoal that can still be useful (for a craft job: its ``use`` is not yet
   credited). If no such job exists the pick is redundant and earns nothing.
2. A successful ``use`` producing O credits the first pending craft job for O without ``use``
   credit. Its input picks must have happened, so they are credited too (see 4).
3. A successful ``drop`` of type T at location L completes the first pending job, in job order,
   whose target is T and destination is L. Completion is **latched** (design decision D4): the job
   stays done even if the item is picked up again later, e.g. by a dependent job that consumes it.
   All of the job's subgoals become credited.
4. When rules 2-3 imply a subgoal the job lacks, the credit is **transferred** from another
   pending job holding that same credit (so the total does not change), or, if no donor exists,
   granted (counted once, emitted as an event with ``inferred=True``).

Transfers never change the count; grants and new credits increase it by one each. Hence the count
is monotone and bounded by ``total_subgoals``.

This module depends only on the data model and recipe book (not on World or the tool layer).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .recipes import crafted_input, recipe_for
from .types import Job, JobKind

PICK = "pick"
USE = "use"
DELIVER = "deliver"


@dataclass(frozen=True)
class Subgoal:
    job_id: str
    kind: str  # PICK | USE | DELIVER
    item_type: str

    @property
    def label(self) -> str:
        return f"{self.job_id}:{self.kind}:{self.item_type}"


@dataclass(frozen=True)
class SubgoalEvent:
    """Emitted each time the subgoal count increases by one."""

    count: int  # subgoals completed after this event
    subgoal: Subgoal
    item_id: str | None
    inferred: bool = False  # credited by rule 4 without a donor

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "job_id": self.subgoal.job_id,
            "kind": self.subgoal.kind,
            "item_type": self.subgoal.item_type,
            "item_id": self.item_id,
            "inferred": self.inferred,
        }


def subgoals_for(job: Job) -> tuple[Subgoal, ...]:
    if job.kind is JobKind.DELIVER:
        return (Subgoal(job.id, PICK, job.target_type), Subgoal(job.id, DELIVER, job.target_type))
    recipe = recipe_for(job.target_type)
    return (
        *(Subgoal(job.id, PICK, t) for t in recipe.inputs),
        Subgoal(job.id, USE, job.target_type),
        Subgoal(job.id, DELIVER, job.target_type),
    )


def count_subgoals(jobs: tuple[Job, ...] | list[Job]) -> int:
    return sum(len(subgoals_for(j)) for j in jobs)


def describe_job(job: Job, jobs_by_id: dict[str, Job] | None = None) -> str:
    """The job text the agent reads (in the task description and in ``status``)."""
    if job.kind is JobKind.DELIVER:
        return f"Deliver a {job.target_type} to {job.destination}."
    r = recipe_for(job.target_type)
    text = (
        f"Make a {job.target_type} at a {r.station_type} from {r.inputs[0]} + {r.inputs[1]}, "
        f"then deliver it to {job.destination}."
    )
    ci = crafted_input(job.target_type)
    if ci is not None and job.depends_on:
        producers = [d for d in job.depends_on if jobs_by_id is None or d in jobs_by_id]
        text += f" Use the {ci} delivered for {', '.join(producers)}."
    return text


class ProgressTracker:
    """Observes successful write actions and tracks subgoal credit and job completion."""

    def __init__(self, jobs: tuple[Job, ...] | list[Job]) -> None:
        self.jobs: tuple[Job, ...] = tuple(jobs)
        self._jobs_by_id = {j.id: j for j in self.jobs}
        self._subgoals: dict[str, tuple[Subgoal, ...]] = {j.id: subgoals_for(j) for j in self.jobs}
        self._credited: set[Subgoal] = set()
        self._done: dict[str, bool] = {j.id: False for j in self.jobs}
        self._count = 0
        self._total = count_subgoals(self.jobs)

    # ------------------------------------------------------------------ queries

    def subgoals_completed(self) -> int:
        return self._count

    def total_subgoals(self) -> int:
        return self._total

    def progress_rate(self) -> float:
        return self._count / self._total if self._total else 1.0

    def job_done(self, job_id: str) -> bool:
        return self._done[job_id]

    def all_done(self) -> bool:
        return all(self._done.values())

    def pending_jobs(self) -> list[Job]:
        return [j for j in self.jobs if not self._done[j.id]]

    def is_credited(self, sg: Subgoal) -> bool:
        return sg in self._credited

    def job_subgoals(self, job_id: str) -> dict[str, bool]:
        return {sg.label: sg in self._credited for sg in self._subgoals[job_id]}

    def jobs_view(self) -> list[dict[str, Any]]:
        """The job list ``status`` shows: ``{"id", "description", "done"}`` in job order."""
        return [
            {"id": j.id, "description": describe_job(j, self._jobs_by_id), "done": self._done[j.id]}
            for j in self.jobs
        ]

    def task_description(self) -> str:
        lines = ["Complete all of these jobs:"]
        lines += [f"- {j.id}: {describe_job(j, self._jobs_by_id)}" for j in self.jobs]
        return "\n".join(lines)

    # ------------------------------------------------------------------ observe

    def observe(self, tool: str, ok: bool, data: dict[str, Any]) -> list[SubgoalEvent]:
        """Update credit from one tool result. Returns the events that increased the count.

        Takes the tool name, ``ok`` flag, and the result ``data`` produced by ``World`` (via
        ``ToolResult.data``). Read tools and failed calls never change progress.
        """
        if not ok:
            return []
        if tool == "pick":
            return self._on_pick(data["type"], data["item"])
        if tool == "use":
            return self._on_use(data["output_type"], data["output"])
        if tool == "drop":
            return self._on_drop(data["type"], data["location"], data["item"])
        return []

    def _on_pick(self, item_type: str, item_id: str) -> list[SubgoalEvent]:
        for job in self.pending_jobs():
            sg = Subgoal(job.id, PICK, item_type)
            if sg in self._subgoals[job.id] and sg not in self._credited and self._pick_useful(job):
                return [self._credit(sg, item_id)]
        return []

    def _on_use(self, output_type: str, output_id: str) -> list[SubgoalEvent]:
        for job in self.pending_jobs():
            if job.kind is not JobKind.CRAFT or job.target_type != output_type:
                continue
            sg = Subgoal(job.id, USE, output_type)
            if sg in self._credited:
                continue
            events: list[SubgoalEvent] = []
            for inp in recipe_for(output_type).inputs:
                events += self._ensure(Subgoal(job.id, PICK, inp), None)
            events.append(self._credit(sg, output_id))
            return events
        return []

    def _on_drop(self, item_type: str, location: str, item_id: str) -> list[SubgoalEvent]:
        for job in self.pending_jobs():
            if job.target_type != item_type or job.destination != location:
                continue
            events: list[SubgoalEvent] = []
            for sg in self._subgoals[job.id]:
                if sg.kind == DELIVER:
                    events.append(self._credit(sg, item_id))
                else:
                    events += self._ensure(sg, None)
            self._done[job.id] = True
            return events
        return []

    # ------------------------------------------------------------------ credit bookkeeping

    def _pick_useful(self, job: Job) -> bool:
        """A craft job's input pick only counts before its station use."""
        if job.kind is JobKind.CRAFT:
            return Subgoal(job.id, USE, job.target_type) not in self._credited
        return True

    def _credit(self, sg: Subgoal, item_id: str | None, inferred: bool = False) -> SubgoalEvent:
        assert sg not in self._credited, sg
        self._credited.add(sg)
        self._count += 1
        return SubgoalEvent(self._count, sg, item_id, inferred)

    def _ensure(self, sg: Subgoal, item_id: str | None) -> list[SubgoalEvent]:
        """Make sure ``sg`` is credited, preferring a transfer from another pending job."""
        if sg in self._credited:
            return []
        donor = self._find_donor(sg)
        if donor is not None:
            self._credited.discard(donor)
            self._credited.add(sg)
            return []
        return [self._credit(sg, item_id, inferred=True)]

    def _find_donor(self, sg: Subgoal) -> Subgoal | None:
        for job in self.pending_jobs():
            if job.id == sg.job_id:
                continue
            cand = Subgoal(job.id, sg.kind, sg.item_type)
            if cand not in self._credited:
                continue
            # Never take a credit the donor has already built on.
            if sg.kind == PICK and not self._pick_useful(job):
                continue
            return cand
        return None

    # ------------------------------------------------------------------ copy / serialize

    def clone(self) -> "ProgressTracker":
        c = ProgressTracker.__new__(ProgressTracker)
        c.jobs = self.jobs
        c._jobs_by_id = self._jobs_by_id
        c._subgoals = self._subgoals
        c._credited = set(self._credited)
        c._done = dict(self._done)
        c._count = self._count
        c._total = self._total
        return c

    def restricted_to(self, job_ids: set[str] | list[str]) -> "ProgressTracker":
        """A clone in which every job *not* in ``job_ids`` counts as done.

        Used to ask the solver about a subset of jobs, e.g. whether a disruption affects one job.
        Subgoal counts of the clone are not meaningful."""
        c = self.clone()
        keep = set(job_ids)
        for j in self.jobs:
            if j.id not in keep:
                c._done[j.id] = True
        return c

    def state_key(self) -> tuple:
        """Hashable progress state (for solver search nodes)."""
        return (
            tuple(sorted(sg.label for sg in self._credited)),
            tuple(self._done[j.id] for j in self.jobs),
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "subgoals_completed": self._count,
            "total_subgoals": self._total,
            "jobs": [
                {"id": j.id, "done": self._done[j.id], "subgoals": self.job_subgoals(j.id)}
                for j in self.jobs
            ],
        }
