"""Rule-based failure categories (FP3 4.6), computed from one episode trace.
 
Exactly one category per failed episode, in this order (FP2 section 4.5):
    stale_plan_execution > goal_drift > budget_exhaustion > other
 
STATUS: all three rules implemented. Stale-plan execution is slightly looser than FP2's wording
(see its docstring), so decide before the pilot whether to tighten it.
"""
from __future__ import annotations
 
from typing import Any
 
IDENTICAL_CALLS = 3          # FP2: the same tool call three times in a row ...
WRITES_WITHOUT_PROGRESS = 10  # ... or ten write actions in a row without completing a subgoal.
CAP_REASONS = ("token_cap", "action_cap", "step_cap")
 
 
def categorize(trace: dict[str, Any]) -> "str | None":
    """None for a success, otherwise one of the four categories."""
    if trace["m5"]["success"]:
        return None
    calls = trace["m2"]
    if _stale_plan(calls, trace["m3"]):
        return "stale_plan_execution"
    if _goal_drift(calls):
        return "goal_drift"
    if trace["m5"]["stop_reason"] in CAP_REASONS:
        return "budget_exhaustion"
    return "other"
 
 
def _goal_drift(calls: list[dict[str, Any]]) -> bool:
    run = 1
    for prev, cur in zip(calls, calls[1:]):
        same = (prev["tool"], prev["args"]) == (cur["tool"], cur["args"])
        run = run + 1 if same else 1
        if run >= IDENTICAL_CALLS:
            return True
    streak, last_progress = 0, 0
    for call in calls:
        if not call["is_write"]:
            continue
        if call["subgoals_completed"] > last_progress:
            streak, last_progress = 0, call["subgoals_completed"]
        else:
            streak += 1
            if streak >= WRITES_WITHOUT_PROGRESS:
                return True
    return False
 
 
def _stale_plan(calls: list[dict[str, Any]], disruptions: list[dict[str, Any]]) -> bool:
    """After a disruption fired, the agent issued a write on the disrupted resource and it failed.
 
    M3 gives each fired disruption's target ("road:a|b", "item:x", "station:s") and M2 gives the
    resources each call touched in the same format, so this is an exact match. The index of the
    call a disruption fired after is in M2 (`disruptions_fired`).
 
    Looser than FP2's wording in one respect: it does not check that an earlier tool result had
    already *revealed* the disruption. A write that walks into an unseen disruption also counts.
    If the group wants the strict reading, add that condition here and report it in 7.1."""
    targets = {d["target"] for d in disruptions if d.get("fired") and d.get("target")}
    first_fired = next((c["index"] for c in calls if c["disruptions_fired"]), None)
    if first_fired is None or not targets:
        return False
    return any(
        c["is_write"] and not c["ok"] and c["index"] > first_fired and targets & set(c["resources"])
        for c in calls
    )