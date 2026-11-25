"""Token caps from the uncapped pilot (FP3 10.7): 1.5x the median ReAct token usage.
 
    python -m runner.derive_caps [--by-cell] [--models gemini-flash-lite ...] [--set pilot]
 
Reads every worker's ReAct file (results/*/<set>__react*.jsonl), so merge everyone's pilot results
into results/ first. Tokens are counted as Budget counts them: uncached input + cached input +
output. Pooled across models by default (FP2: "set separately for each length"); --by-cell keys by
length and disruption count, which D8 in errandworld/ARCHITECTURE.md argues for. Writes
configs/caps.json.
"""
from __future__ import annotations
 
import argparse
import json
import math
import statistics
from collections import defaultdict
 
from runner.run import RESULTS, ROOT
 
 
def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--by-cell", action="store_true")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--set", dest="set_name", default="pilot")
    args = ap.parse_args(argv)
 
    usage: dict[str, list[int]] = defaultdict(list)
    seen: set[tuple] = set()
    for path in sorted(RESULTS.glob(f"*/{args.set_name}__react*.jsonl")):
        if args.models and path.parent.name not in args.models:
            continue
        for line in path.read_text().splitlines():
            r = json.loads(line)
            key = (r["scenario_id"], r["model"], r["run"])
            if r.get("infrastructure_failure") or key in seen:  # an episode two people both ran counts once
                continue
            seen.add(key)
            cell = f"L{r['length_level']}" + (f"_D{r['disruption_level']}" if args.by_cell else "")
            usage[cell].append(r["input_tokens"] + r["cached_input_tokens"] + r["output_tokens"])
    if not usage:
        raise SystemExit(f"no ReAct records under results/*/{args.set_name}__react*.jsonl")
    caps = {key: math.ceil(1.5 * statistics.median(v)) for key, v in sorted(usage.items())}
    note = (f"1.5 x median ReAct {args.set_name} tokens, {'per cell' if args.by_cell else 'per length'}; n per key: "
            + ", ".join(f"{k}={len(v)}" for k, v in sorted(usage.items())))
    (ROOT / "configs" / "caps.json").write_text(json.dumps({"_note": note, **caps}, indent=2) + "\n")
    print(note)
    print(json.dumps(caps, indent=2))
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())