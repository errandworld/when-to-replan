from metrics.bootstrap import paired_delta
from metrics.cost import cost_per_success, episode_cost
from metrics.failure import categorize
 
 
def rec(sid, strategy, success, **kw):
    return {"scenario_id": sid, "strategy": strategy, "success": success, **kw}
 
 
def test_delta_is_paired_by_scenario():
    rows = [rec(f"s{i}", "adaptive", 1) for i in range(20)] + [rec(f"s{i}", "react", int(i % 2)) for i in range(20)]
    d = paired_delta(rows, n_boot=500)
    assert d["n"] == 20 and abs(d["delta"] - 0.5) < 1e-9
    assert d["ci95"][0] <= d["delta"] <= d["ci95"][1]
 
 
def test_infrastructure_failures_and_unpaired_scenarios_are_left_out():
    rows = [rec("a", "adaptive", 1), rec("a", "react", 0), rec("b", "adaptive", 1),
            rec("c", "adaptive", 0, infrastructure_failure=True), rec("c", "react", 1)]
    assert paired_delta(rows, n_boot=100)["n"] == 1
 
 
def test_cost_prices_cached_and_uncached_input_separately_and_undefined_without_successes():
    prices = {"models": {"m": {"input": 1.0, "cached_input": 0.1, "output": 2.0}}}
    r = {"model": "m", "input_tokens": 1_000_000, "cached_input_tokens": 1_000_000, "output_tokens": 500_000, "success": True}
    assert episode_cost(r, prices) == 1.0 + 0.1 + 1.0
    assert cost_per_success([r, {**r, "success": False}], prices) == (2.1 * 2) / 1
    assert cost_per_success([{**r, "success": False}], prices) is None
    assert episode_cost(r, {"models": {"m": {"input": None, "cached_input": None, "output": None}}}) is None
 
 
def call(i, tool="move", ok=True, write=True, progress=0, args=None, error=None, resources=(), fired=()):
    return {"index": i, "tool": tool, "args": args or {"location": f"x{i}"}, "ok": ok, "error": error, "is_write": write,
            "subgoals_completed": progress, "resources": list(resources), "disruptions_fired": list(fired)}
 
 
def trace(calls, stop="agent_stopped", success=False, disruptions=()):
    return {"m2": calls, "m3": list(disruptions), "m5": {"success": success, "stop_reason": stop}}
 
 
def test_failure_categories_in_the_fixed_order():
    assert categorize(trace([], success=True)) is None
    spin = [call(i, args={"location": "a"}) for i in range(3)]
    assert categorize(trace(spin)) == "goal_drift"
    assert categorize(trace([call(i) for i in range(10)])) == "goal_drift"
    assert categorize(trace([call(0), call(1)], stop="action_cap")) == "budget_exhaustion"
    assert categorize(trace([call(0), call(1)])) == "other"
    stale = [call(0, fired=[0]), call(1, ok=False, error="road_closed", resources=["road:a|b"])]
    d = [{"fired": True, "target": "road:a|b"}]
    assert categorize(trace(stale, stop="action_cap", disruptions=d)) == "stale_plan_execution"  # outranks budget