"""Local repair's cap: the solver minimum for one broken step.
 
FP2 caps a repair at "solver minimum + 2 write actions". The minimum comes from hidden state (the
true roads, stations and item positions), which the agent loops must never see, so the harness
computes it from its shadow Episode and hands the loop a callback (`make_repair_limit`). The loop
only ever receives a number, and truncates the model's repair afterwards instead of telling the
model (Part A doc, requirement 7: stating it in the prompt could leak a silent disruption).
 
What "the minimum" means for each step (document this in 9.1 and FP3 8.5):
 
  move X     moves along open roads from here to X.
  pick I     moves to where I lies, plus the pick.
  drop I     the drop itself (1).
  use S ...  moves to the nearest *online* station of S's type, plus the use. It assumes the
             inputs are already carried; if they are not, or S is unknown, there is no cap (None).
             It does not count the walk back to S's old location. The "+2" slack in FP2's rule
             is what pays for that, so a repair that has to return costs the model its slack.
"""
from __future__ import annotations
 
from collections import deque
from typing import Any, Callable
 
from app.checker import Trigger
from app.plan import PlanStep
from errandworld.types import road_key
 
 
def make_repair_limit(shadow: Any) -> Callable[[PlanStep, Trigger], "int | None"]:
    """The `repair_limit` callback for run_local_repair, reading `shadow`'s true world."""
 
    def limit(step: PlanStep, trigger: Trigger) -> "int | None":
        return repair_min(shadow.world, step)
 
    return limit
 
 
def repair_min(world: Any, step: PlanStep) -> "int | None":
    """Fewest write actions that achieve what `step` set out to do, from the true current state.
    None when it cannot be computed (no cap is applied)."""
    here = world.agent_location
    args = step.args
    if step.tool == "move":
        return _distance(world, here, args.get("location"))
    if step.tool == "pick":
        item = args.get("item")
        if item in world.inventory:
            return 0
        where = world.item_location.get(item)
        steps = _distance(world, here, where)
        return None if steps is None else steps + 1
    if step.tool == "drop":
        return 1
    if step.tool == "use":
        return _use_cost(world, here, args)
    return None  # look, status, read_map are reads: they are not write actions and have no cap
 
 
def _use_cost(world: Any, here: str, args: dict[str, Any]) -> "int | None":
    station, items = args.get("station"), args.get("items") or []
    if station not in world.station_type or any(item not in world.inventory for item in items):
        return None
    kind = world.station_type[station]
    distances = [
        _distance(world, here, world.station_location[other])
        for other, other_kind in world.station_type.items()
        if other_kind == kind and world.station_online[other]
    ]
    reachable = [d for d in distances if d is not None]
    return min(reachable) + 1 if reachable else None
 
 
def _distance(world: Any, source: str, target: Any) -> "int | None":
    """Fewest moves from source to target over roads that are open right now."""
    if target is None or target not in world.location_names:
        return None
    if source == target:
        return 0
    seen, queue = {source}, deque([(source, 0)])
    while queue:
        node, dist = queue.popleft()
        for nxt in world.neighbors(node):
            if nxt in seen or not world.roads_open[road_key(node, nxt)]:
                continue
            if nxt == target:
                return dist + 1
            seen.add(nxt)
            queue.append((nxt, dist + 1))
    return None