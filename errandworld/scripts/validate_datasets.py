"""Validate the generated scenario sets (Phase 1 exit criteria, FP2 section 4.4).

    python scripts/validate_datasets.py --set all --workers 6

For every scenario:
  1. schema      - the scenario file passes integrity checks; its map passes map invariants
                   (connected, no bridges, two stations per type at distinct locations).
  2. solver      - solvable from the start, optimum proven, equal to the recorded
                   optimal_actions, and within +/-10% of the length level.
  3. replay      - along the reference trajectory (optimal plan, re-planned after each
                   disruption) every disruption fires, raises the optimal cost, and affects at
                   least one pending job; targets match those recorded at generation time.
  4. oracle      - the full-state oracle, playing through Episode.call, succeeds within the action
                   cap (2 x reference_actions + 5) with every disruption fired strictly, using
                   exactly reference_actions writes (the optimum with disruptions).

Writes data/validation/<set>_report.json (per-scenario results + per-cell summary) and prints a
summary table. Exits with status 1 if any scenario fails any check.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from errandworld.disruptions import all_fired_strictly, replay_optimal  # noqa: E402
from errandworld.generator import DISRUPTION_LEVELS, LENGTH_LEVELS, GeneratorConfig  # noqa: E402
from errandworld.maps import map_invariant_problems  # noqa: E402
from errandworld.oracle import run_oracle  # noqa: E402
from errandworld.progress import ProgressTracker  # noqa: E402
from errandworld.scenario_io import load_scenario, scenario_problems  # noqa: E402
from errandworld.solver import solve  # noqa: E402
from errandworld.world import World  # noqa: E402

SCEN_DIR = ROOT / "data" / "scenarios"
OUT_DIR = ROOT / "data" / "validation"
SETS = ("dev", "pilot", "test")


def validate_one(path: str) -> dict:
    t0 = time.perf_counter()
    s = load_scenario(path)
    problems: list[str] = []

    # 1. schema
    problems += [f"schema: {p}" for p in scenario_problems(s)]
    problems += [f"map: {p}" for p in map_invariant_problems(s.map)]

    # 2. solver
    res = solve(World.from_scenario(s), ProgressTracker(s.jobs))
    lo, hi = GeneratorConfig().band(s.length_level)
    if not res.solvable:
        problems.append("solver: not solvable")
    elif not res.proven_optimal:
        problems.append("solver: optimum not proven")
    else:
        if res.cost != s.optimal_actions:
            problems.append(f"solver: optimum {res.cost} != recorded {s.optimal_actions}")
        if not lo <= res.cost <= hi:
            problems.append(f"solver: optimum {res.cost} outside [{lo}, {hi}]")

    # 3. schedule replay
    events, replay_steps = replay_optimal(s)
    if not all_fired_strictly(events):
        bad = [(e.index, e.type.value, e.reason or "fallback") for e in events
               if not (e.fired and not e.fallback)]
        problems.append(f"replay: not all disruptions fired strictly: {bad}")
    if any(e.fired and not e.affected_jobs for e in events):
        problems.append("replay: a disruption affected no job")
    recorded = [e["target"] for e in s.metadata.get("reference_replay", [])]
    if [e.target for e in events] != recorded:
        problems.append(f"replay: targets {[e.target for e in events]} != recorded {recorded}")

    # 4. oracle
    ep = run_oracle(s)
    sm = ep.summary()
    if not ep.is_success():
        problems.append(f"oracle: ended with {ep.end_reason}")
    if sm["disruptions_fired"] != len(s.schedule) or sm["disruptions_fallback"]:
        problems.append(
            f"oracle: fired {sm['disruptions_fired']}/{len(s.schedule)}, "
            f"fallback {sm['disruptions_fallback']}"
        )
    if [e.target for e in ep.disruption_events] != recorded:
        problems.append("oracle: disruption targets differ from the reference replay")
    if s.reference_actions is None:
        problems.append("schema: reference_actions missing")
    elif sm["write_calls"] != s.reference_actions or replay_steps != s.reference_actions:
        problems.append(f"oracle: {sm['write_calls']} writes (replay {replay_steps}) != "
                        f"reference_actions {s.reference_actions}")

    return {
        "id": s.id,
        "length_level": s.length_level,
        "disruption_level": s.disruption_level,
        "ok": not problems,
        "problems": problems,
        "optimal_actions": s.optimal_actions,
        "oracle_writes": sm["write_calls"],
        "oracle_extra_writes": sm["write_calls"] - s.optimal_actions,
        "action_cap": ep.action_cap,
        "cap_headroom": ep.action_cap - sm["write_calls"],
        "oracle_replans": getattr(ep, "oracle_replans", None),
        "replay_writes": replay_steps,
        "disruption_cost_increase": [e.cost_after - e.cost_before for e in events if e.fired],
        "injector_seconds": sm["injector_seconds"],
        "seconds": round(time.perf_counter() - t0, 3),
    }


def summarize(results: list[dict]) -> dict:
    cells = {}
    for L in LENGTH_LEVELS:
        for D in DISRUPTION_LEVELS:
            g = [r for r in results if r["length_level"] == L and r["disruption_level"] == D]
            if not g:
                continue
            extra = [r["oracle_extra_writes"] for r in g]
            incr = [x for r in g for x in r["disruption_cost_increase"]]
            cells[f"L{L}_D{D}"] = {
                "n": len(g),
                "passed": sum(r["ok"] for r in g),
                "oracle_success": sum(not any(p.startswith("oracle: ended") for p in r["problems"])
                                      for r in g),
                "oracle_extra_writes_mean": round(statistics.mean(extra), 2),
                "oracle_extra_writes_max": max(extra),
                "min_cap_headroom": min(r["cap_headroom"] for r in g),
                "disruption_cost_increase_mean": round(statistics.mean(incr), 2) if incr else None,
            }
    return cells


def run_set(set_name: str, workers: int) -> tuple[bool, dict]:
    files = sorted(str(p) for p in (SCEN_DIR / set_name).glob(f"{set_name}_L*_D*_*.json"))
    if not files:
        print(f"{set_name}: no scenarios found")
        return False, {}
    t0 = time.perf_counter()
    if workers > 1:
        with Pool(workers) as pool:
            results = pool.map(validate_one, files)
    else:
        results = [validate_one(f) for f in files]
    results.sort(key=lambda r: r["id"])
    report = {
        "set": set_name,
        "n_scenarios": len(results),
        "n_passed": sum(r["ok"] for r in results),
        "seconds": round(time.perf_counter() - t0, 1),
        "cells": summarize(results),
        "failures": [r for r in results if not r["ok"]],
        "scenarios": results,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{set_name}_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report["n_passed"] == len(results), report


def print_report(rep: dict) -> None:
    print(f"\n{rep['set']}: {rep['n_passed']}/{rep['n_scenarios']} passed in {rep['seconds']}s")
    print(f"{'cell':<8} {'pass':>7} {'oracle ok':>9} {'extra writes':>13} {'min headroom':>12} "
          f"{'disr. +cost':>11}")
    for cell, c in rep["cells"].items():
        print(f"{cell:<8} {c['passed']:>3}/{c['n']:<3} {c['oracle_success']:>5}/{c['n']:<3} "
              f"{c['oracle_extra_writes_mean']:>7} (max {c['oracle_extra_writes_max']:>2}) "
              f"{c['min_cap_headroom']:>12} {str(c['disruption_cost_increase_mean']):>11}")
    for f in rep["failures"][:20]:
        print(f"  FAIL {f['id']}: {f['problems']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=[*SETS, "all"], required=True)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    ok_all = True
    for name in (SETS if a.set == "all" else [a.set]):
        ok, rep = run_set(name, a.workers)
        if rep:
            print_report(rep)
        ok_all &= ok
    print("\nALL CHECKS PASSED" if ok_all else "\nVALIDATION FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
