"""Oracle agent: a non-LLM agent with full access to the true world state.

It follows an optimal plan from the exact solver and re-plans from the current true state whenever
a call fails or a disruption fires. It plays through ``Episode.call`` exactly like an LLM agent
would, so it exercises the real episode path (action cap, disruption firing, logging).

FP2 section 4.4 requires the oracle to succeed on 100% of scenarios within both caps: that shows
every failure in the experiment comes from the agent, not from the environment or the caps.
"""

from __future__ import annotations

from .episode import RUNNING, Episode
from .solver import SolverConfig, plan
from .types import Scenario


def run_oracle(
    scenario: Scenario,
    *,
    action_cap: int | None = None,
    solver_config: SolverConfig | None = None,
) -> Episode:
    """Play one episode with the oracle and return the finished Episode."""
    ep = Episode(scenario, action_cap=action_cap, solver_config=solver_config)
    calls = plan(ep.world, ep.tracker, solver_config) or []
    replans = 0
    while ep.status == RUNNING:
        if not calls:
            ep.end("oracle_no_plan")
            break
        name, args = calls.pop(0)
        result = ep.call(name, args)
        if not result.ok or ep.tool_calls[-1].disruptions_fired:
            calls = plan(ep.world, ep.tracker, solver_config) or []
            replans += 1
    ep.oracle_replans = replans  # type: ignore[attr-defined]
    return ep
