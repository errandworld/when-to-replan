"""Per-cell summary of results: success rate, replanning advantage, cost.
 
    python -m metrics.summarize results/gemini-flash-lite --set pilot
    python -m metrics.summarize results/*  --set pilot        # several models pooled
 
Reads every worker's JSONL file in the folders. An episode that two workers both ran is counted
once (the first copy) and reported, so merged results never double-count.
"""
from __future__ import annotations
 
import argparse
import json
from collections import defaultdict
from pathlib import Path
 
from metrics.bootstrap import paired_delta
from metrics.cost import cost_per_success, load_prices
 
 
def load_records(folders: list[Path], set_name: "str | None") -> "tuple[list[dict], int]":
    records, seen, duplicates = [], set(), 0
    for folder in folders:
        for path in sorted(folder.glob("*.jsonl")):
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if set_name and r.get("set") != set_name:
                    continue
                key = (r["scenario_id"], r["strategy"], r["model"], r["run"])
                if key in seen:
                    duplicates += 1
                    continue
                seen.add(key)
                records.append(r)
    return records, duplicates
 
 
def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+", type=Path)
    ap.add_argument("--set", dest="set_name", default=None, help="dev, pilot or test (default: all, which mixes them)")
    args = ap.parse_args(argv)
 
    records, duplicates = load_records(args.folders, args.set_name)
    if not args.set_name and len({r.get("set") for r in records}) > 1:
        print("warning: records from more than one set are mixed together; pass --set\n")
    if duplicates:
        print(f"note: {duplicates} duplicate episode(s) ignored (the same four-tuple in two files)\n")
    prices = load_prices()
    cells: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for r in records:
        cells[(r["length_level"], r["disruption_level"])].append(r)
    print(f"{'cell':8} {'strategy':13} {'n':>3} {'success':>8} {'infra':>6} {'$/success':>10}")
    for (length, disruptions), rs in sorted(cells.items()):
        for strategy in ("react", "adaptive", "local_repair"):
            sub = [r for r in rs if r["strategy"] == strategy]
            if not sub:
                continue
            ok = [r for r in sub if not r.get("infrastructure_failure")]
            rate = sum(bool(r["success"]) for r in ok) / len(ok) if ok else float("nan")
            cps = cost_per_success(ok, prices)
            print(f"L{length}_D{disruptions:<4} {strategy:13} {len(ok):>3} {rate:>8.2f} "
                  f"{len(sub) - len(ok):>6} {'undefined' if cps is None else f'{cps:.4f}':>10}")
        d = paired_delta(rs)
        if d["delta"] is not None:
            print(f"{'':8} delta(adaptive-react) = {d['delta']:+.2f}  95% CI [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]  (n={d['n']})")
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())