"""How often local repair's write cap bites, per cell.

    python -m metrics.repair_stats results/gemini-flash-lite --set pilot
    python -m metrics.repair_stats results/* --set pilot        # several models pooled

A repair may take the solver's minimum plus two write actions (app/repair.py). The cap comes
from hidden world state, so the model is never told it: a longer repair is cut after the reply.
This report keeps that truncation visible. Per cell, over local-repair episodes that finished
without an infrastructure failure:

    repairs        repairs made
    capped         repairs the cap applied to (no cap when the solver minimum is undefined)
    truncated      capped repairs that the cap cut short
    cut share      truncated / capped
    steps cut      plan steps removed by the cap, reads after the cut included
    episodes cut   episodes with at least one truncated repair

Reads every worker's JSONL file in the folders, counting an episode two workers both ran once.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from metrics.summarize import load_records

FIELDS = ("revisions", "repairs_capped", "repairs_truncated", "steps_cut")


def repair_rows(records: list[dict[str, Any]]) -> "tuple[list[dict[str, Any]], int]":
    """One row per cell in (length, disruptions) order, then an "all" row. Also returns how
    many local-repair records were left out because they predate the truncation counts."""
    cells: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    missing = 0
    for r in records:
        if r.get("strategy") != "local_repair" or r.get("infrastructure_failure"):
            continue
        if any(name not in r for name in FIELDS):
            missing += 1
            continue
        cells[(r["length_level"], r["disruption_level"])].append(r)
    rows = [_row(f"L{length}_D{disruptions}", rs) for (length, disruptions), rs in sorted(cells.items())]
    if rows:
        rows.append(_row("all", [r for rs in cells.values() for r in rs]))
    return rows, missing


def _row(cell: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    capped = sum(r["repairs_capped"] for r in records)
    truncated = sum(r["repairs_truncated"] for r in records)
    return {
        "cell": cell,
        "episodes": len(records),
        "repairs": sum(r["revisions"] for r in records),
        "capped": capped,
        "truncated": truncated,
        "cut_share": truncated / capped if capped else None,
        "steps_cut": sum(r["steps_cut"] for r in records),
        "episodes_cut": sum(r["repairs_truncated"] > 0 for r in records),
    }


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+", type=Path)
    ap.add_argument("--set", dest="set_name", default=None, help="dev, pilot or test (default: all, which mixes them)")
    args = ap.parse_args(argv)

    records, duplicates = load_records(args.folders, args.set_name)
    rows, missing = repair_rows(records)
    if duplicates:
        print(f"note: {duplicates} duplicate episode(s) ignored (the same four-tuple in two files)")
    if missing:
        print(f"note: {missing} local-repair record(s) predate the truncation counts and are left out")
    if not rows:
        print("no local-repair records with truncation counts")
        return 0
    print(f"{'cell':8} {'episodes':>8} {'repairs':>7} {'capped':>6} {'truncated':>9} "
          f"{'cut share':>9} {'steps cut':>9} {'episodes cut':>12}")
    for row in rows:
        share = "undefined" if row["cut_share"] is None else f"{row['cut_share']:.2f}"
        print(f"{row['cell']:8} {row['episodes']:>8} {row['repairs']:>7} {row['capped']:>6} "
              f"{row['truncated']:>9} {share:>9} {row['steps_cut']:>9} {row['episodes_cut']:>12}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
