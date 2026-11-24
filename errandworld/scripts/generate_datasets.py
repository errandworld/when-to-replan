"""Generate the ErrandWorld scenario sets.

    python scripts/generate_datasets.py --set dev            # 45 scenarios (5 per cell)
    python scripts/generate_datasets.py --set pilot          # 45 scenarios (5 per cell)
    python scripts/generate_datasets.py --set test           # 270 scenarios (30 per cell)
    python scripts/generate_datasets.py --set all --workers 8
    python scripts/generate_datasets.py --verify --set all   # re-check files on disk, write nothing

A cell is one (length level, disruption level) pair: 3 x 3 = 9 cells. Within a cell, scenario i
uses map ``maps[i % len(maps)]``, composition index i, and seed stable_seed(set, L, D, i), so the
output is identical regardless of worker count or order. Disruption types are balanced within
each cell (counts differ by at most one) and distinct within each scenario.

Outputs: data/scenarios/<set>/<scenario_id>.json and data/scenarios/<set>/manifest.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from errandworld.generator import (  # noqa: E402
    DISRUPTION_LEVELS,
    GENERATOR_VERSION,
    LENGTH_LEVELS,
    GeneratorConfig,
    cell_disruption_types,
    generate_scenario,
    stable_seed,
)
from errandworld.scenario_io import (  # noqa: E402
    canonical_json,
    check_scenario,
    fingerprint,
    load_map,
    load_scenario,
    save_scenario,
)
from errandworld.types import DisruptionType  # noqa: E402

MAP_DIR = ROOT / "data" / "maps"
OUT_DIR = ROOT / "data" / "scenarios"

SETS: dict[str, dict] = {
    "dev": {"maps": ["dev_map_1", "dev_map_2"], "per_cell": 5},
    "pilot": {"maps": [f"test_map_{i}" for i in range(1, 6)], "per_cell": 5},
    "test": {"maps": [f"test_map_{i}" for i in range(1, 6)], "per_cell": 30},
}


def scenario_id(set_name: str, L: int, D: int, i: int) -> str:
    return f"{set_name}_L{L}_D{D}_{i:03d}"


def plan_set(set_name: str) -> list[dict]:
    """Deterministic list of generation tasks for one set."""
    spec = SETS[set_name]
    tasks = []
    for L in LENGTH_LEVELS:
        for D in DISRUPTION_LEVELS:
            n = spec["per_cell"]
            per_scenario = cell_disruption_types(n, D, stable_seed(set_name, L, D, "types"))
            for i in range(n):
                tasks.append(
                    {
                        "set": set_name,
                        "L": L,
                        "D": D,
                        "i": i,
                        "map": spec["maps"][i % len(spec["maps"])],
                        "types": [t.value for t in per_scenario[i]],
                    }
                )
    return tasks


def run_task(task: dict) -> tuple[dict, object]:
    m = load_map(MAP_DIR / f"{task['map']}.json")
    t0 = time.perf_counter()
    s = generate_scenario(
        m,
        scenario_id=scenario_id(task["set"], task["L"], task["D"], task["i"]),
        set_name=task["set"],
        length_level=task["L"],
        disruption_types=[DisruptionType(t) for t in task["types"]],
        seed=stable_seed(task["set"], task["L"], task["D"], task["i"]),
        index=task["i"],
        config=GeneratorConfig(),
    )
    return {**task, "seconds": time.perf_counter() - t0}, s


def content_key(s) -> str:
    """Hash of what the agent faces, ignoring id/seed/set (to detect duplicates across sets)."""
    d = {
        "map": s.map.id,
        "start": s.start_location,
        "items": sorted((i.type, i.location) for i in s.items),
        "jobs": sorted((j.kind.value, j.target_type, j.destination) for j in s.jobs),
    }
    return hashlib.sha256(canonical_json(d).encode()).hexdigest()[:16]


def summarize(set_name: str, scenarios: list, timings: dict[str, float]) -> dict:
    cells = {}
    for L in LENGTH_LEVELS:
        for D in DISRUPTION_LEVELS:
            group = [s for s in scenarios if s.length_level == L and s.disruption_level == D]
            if not group:
                continue
            comp = [s.metadata["composition"] for s in group]
            rej: dict[str, int] = {}
            for s in group:
                for r, c in s.metadata["rejections"].items():
                    rej[r] = rej.get(r, 0) + c
            types: dict[str, int] = {}
            for s in group:
                for d in s.schedule:
                    types[d.type.value] = types.get(d.type.value, 0) + 1
            lengths = [s.optimal_actions for s in group]
            cells[f"L{L}_D{D}"] = {
                "n": len(group),
                "optimal_actions": {
                    "min": min(lengths), "median": statistics.median(lengths),
                    "max": max(lengths), "mean": round(statistics.mean(lengths), 2),
                },
                "craft_share": round(sum(c["craft"] for c in comp) / sum(c["jobs"] for c in comp), 3),
                "dependent_jobs": sum(c["dependent"] for c in comp),
                "attempts_total": sum(s.metadata["attempt"] + 1 for s in group),
                "rejections": dict(sorted(rej.items())),
                "disruption_types": dict(sorted(types.items())),
                "maps": dict(sorted(
                    {m: sum(s.map.id == m for s in group) for m in {s.map.id for s in group}}.items()
                )),
            }
    all_types: dict[str, int] = {}
    for s in scenarios:
        for d in s.schedule:
            all_types[d.type.value] = all_types.get(d.type.value, 0) + 1
    return {
        "set": set_name,
        "generator_version": GENERATOR_VERSION,
        "n_scenarios": len(scenarios),
        "disruption_types": dict(sorted(all_types.items())),
        "cells": cells,
        "generation_seconds": round(sum(timings.values()), 1),
        "scenarios": {
            s.id: {"fingerprint": fingerprint(s), "content_key": content_key(s)}
            for s in sorted(scenarios, key=lambda x: x.id)
        },
    }


def generate(set_name: str, workers: int) -> dict:
    tasks = plan_set(set_name)
    out = OUT_DIR / set_name
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob(f"{set_name}_L*_D*_*.json"):  # only files this script owns
        old.unlink()
    scenarios, timings = [], {}
    t0 = time.perf_counter()
    if workers > 1:
        with Pool(workers) as pool:
            results = pool.imap_unordered(run_task, tasks)
            for k, (info, s) in enumerate(results, 1):
                save_scenario(s, out / f"{s.id}.json")
                scenarios.append(s)
                timings[s.id] = info["seconds"]
                print(f"\r{set_name}: {k}/{len(tasks)}", end="", flush=True)
    else:
        for k, task in enumerate(tasks, 1):
            info, s = run_task(task)
            save_scenario(s, out / f"{s.id}.json")
            scenarios.append(s)
            timings[s.id] = info["seconds"]
            print(f"\r{set_name}: {k}/{len(tasks)}", end="", flush=True)
    print(f"  done in {time.perf_counter() - t0:.1f}s")
    manifest = summarize(set_name, scenarios, timings)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify(set_name: str) -> list[str]:
    """Re-check files on disk: valid, in band, matching the manifest, planned ids all present."""
    problems: list[str] = []
    out = OUT_DIR / set_name
    manifest_path = out / "manifest.json"
    if not manifest_path.exists():
        return [f"{set_name}: no manifest"]
    manifest = json.loads(manifest_path.read_text())
    cfg = GeneratorConfig()
    expected = {scenario_id(t["set"], t["L"], t["D"], t["i"]) for t in plan_set(set_name)}
    found = {p.stem for p in out.glob(f"{set_name}_L*_D*_*.json")}
    if expected != found:
        problems.append(f"{set_name}: missing {sorted(expected - found)[:5]} extra {sorted(found - expected)[:5]}")
    for sid in sorted(found & expected):
        s = load_scenario(out / f"{sid}.json")
        try:
            check_scenario(s)
        except Exception as e:  # noqa: BLE001
            problems.append(f"{sid}: {e}")
            continue
        lo, hi = cfg.band(s.length_level)
        if not lo <= s.optimal_actions <= hi:
            problems.append(f"{sid}: optimal_actions {s.optimal_actions} outside [{lo}, {hi}]")
        if manifest["scenarios"].get(sid, {}).get("fingerprint") != fingerprint(s):
            problems.append(f"{sid}: fingerprint does not match manifest")
    return problems


def print_summary(m: dict) -> None:
    print(f"\n{m['set']}: {m['n_scenarios']} scenarios, disruption types {m['disruption_types']}")
    print(f"{'cell':<8} {'n':>3} {'opt min/med/max':>16} {'craft':>6} {'attempts':>8}  rejections")
    for cell, c in m["cells"].items():
        o = c["optimal_actions"]
        print(f"{cell:<8} {c['n']:>3} {o['min']:>5}/{o['median']:>4}/{o['max']:>4} "
              f"{c['craft_share']:>6} {c['attempts_total']:>8}  {c['rejections']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=[*SETS, "all"], required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    names = list(SETS) if a.set == "all" else [a.set]

    if a.verify:
        problems = [p for n in names for p in verify(n)]
        print("\n".join(problems) or f"verified: {', '.join(names)}")
        return 1 if problems else 0

    manifests = [generate(n, a.workers) for n in names]
    for m in manifests:
        print_summary(m)

    # No scenario may appear in two sets (dev/pilot/test must be disjoint).
    seen: dict[str, str] = {}
    for n in SETS:
        mp = OUT_DIR / n / "manifest.json"
        if not mp.exists():
            continue
        for sid, info in json.loads(mp.read_text())["scenarios"].items():
            if info["content_key"] in seen:
                print(f"DUPLICATE: {sid} == {seen[info['content_key']]}")
                return 1
            seen[info["content_key"]] = sid
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
