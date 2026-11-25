"""The replanning advantage and its bootstrap interval (FP3 4.3).
 
Delta = success(adaptive) - success(react), paired by scenario, per cell. The interval comes from
resampling *scenarios* with replacement (10,000 times), so every scenario's paired runs, models
included, stay together.
"""
from __future__ import annotations
 
import random
from collections import defaultdict
from typing import Any, Iterable
 
N_BOOT = 10_000
 
 
def paired_delta(
    records: Iterable[dict[str, Any]], a: str = "adaptive", b: str = "react",
    n_boot: int = N_BOOT, seed: int = 0,
) -> dict[str, Any]:
    """Records are run records (the JSONL lines) from ONE cell. Infrastructure failures are
    excluded; a scenario missing either strategy is dropped, so the comparison stays paired."""
    per: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if not r.get("infrastructure_failure"):
            per[r["scenario_id"]][r["strategy"]].append(int(r["success"]))
    diffs = [
        sum(s[a]) / len(s[a]) - sum(s[b]) / len(s[b])
        for s in per.values() if s.get(a) and s.get(b)
    ]
    if not diffs:
        return {"n": 0, "delta": None, "ci95": None}
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return {"n": n, "delta": sum(diffs) / n, "ci95": (means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1])}