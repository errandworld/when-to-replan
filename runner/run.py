"""The experiment runner (FP3 section 3).
 
    python -m runner.run --set pilot --model gemini-flash-lite --per-cell 2 --uncapped
    python -m runner.run --set test --model gemini-flash-lite --runs 1 --shuffle      # FP4
 
Choosing what to run (no code edits between conditions, FP3 2.1 and 2.2):
    --set dev|pilot|test   --model KEY...   --strategy react adaptive local_repair
    --lengths 10 20        only these task lengths
    --disruptions 0 3      only these disruption counts
    --per-cell K           the first K scenarios of each cell
    --runs N               N full passes; pass 1 finishes before pass 2 starts (3.3)
 
Sharing the work between people on different machines:
    --shard 2/3            this worker takes every third scenario, starting at the second. All
                           strategies of a scenario stay with the same worker, so paired
                           comparisons are never split.
    --worker NAME          writes results/<model>/<set>__<strategy>__NAME.jsonl, so two people
                           never edit the same file and git merges cleanly.
 
Bookkeeping: episodes are named by (scenario_id, strategy, model, run) (3.2); one JSONL per
condition under a per-model folder (3.4); finished tuples are skipped on restart (3.5); every
record carries the git commit it was produced by; `--shuffle` randomises order (3.7);
`--dry-run` lists the work and calls nothing.
"""
from __future__ import annotations
 
import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable
 
ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "errandworld" / "data" / "scenarios"
RESULTS = ROOT / "results"
MAX_RERUNS = 2  # infrastructure failures are rerun from the start, and the count is recorded (4.7)
 
 
def load_json(name: str) -> dict[str, Any]:
    return json.loads((ROOT / "configs" / name).read_text())
 
 
def parse_id(scenario_id: str) -> tuple[int, int]:
    """dev_L10_D3_002 -> (10, 3): the task length and the disruption count."""
    _, length, disruptions, _ = scenario_id.split("_")
    return int(length[1:]), int(disruptions[1:])
 
 
def scenario_files(
    set_name: str,
    per_cell: "int | None" = None,
    lengths: "list[int] | None" = None,
    disruptions: "list[int] | None" = None,
    shard: "tuple[int, int] | None" = None,
) -> dict[str, Path]:
    """Scenario id -> file, in id order. Filters apply in this order: lengths and disruptions,
    then the first `per_cell` of each remaining cell, then the shard."""
    found = {p.stem: p for p in sorted((SCENARIOS / set_name).glob("*.json")) if p.stem != "manifest"}
    found = {
        sid: path for sid, path in found.items()
        if (not lengths or parse_id(sid)[0] in lengths) and (not disruptions or parse_id(sid)[1] in disruptions)
    }
    if per_cell is not None:
        taken: dict[tuple[int, int], int] = {}
        kept = {}
        for sid, path in found.items():
            cell = parse_id(sid)
            if taken.get(cell, 0) < per_cell:
                taken[cell] = taken.get(cell, 0) + 1
                kept[sid] = path
        found = kept
    if shard is not None:
        index, count = shard
        found = {sid: found[sid] for position, sid in enumerate(sorted(found)) if position % count == index - 1}
    return found
 
 
def parse_shard(text: str) -> tuple[int, int]:
    try:
        index, count = (int(part) for part in text.split("/"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"--shard wants I/N such as 2/3, got {text!r}") from None
    if not 1 <= index <= count:
        raise argparse.ArgumentTypeError(f"--shard {text}: I must be between 1 and N")
    return index, count
 
 
def token_cap(caps: dict[str, Any], scenario_id: str) -> "int | None":
    length, disruptions = parse_id(scenario_id)
    return caps.get(f"L{length}_D{disruptions}", caps.get(f"L{length}"))
 
 
def git_state() -> "tuple[str, bool | None]":
    """(commit, has uncommitted changes). ("unknown", None) outside a git checkout."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip())
    except Exception:
        return "unknown", None
    return (commit, dirty) if commit else ("unknown", None)
 
 
def existing_records(model: str) -> list[dict[str, Any]]:
    """Every record in the model's folder, from every worker's file."""
    return [json.loads(line) for path in sorted((RESULTS / model).glob("*.jsonl")) for line in path.read_text().splitlines() if line.strip()]
 
 
def done_keys(model: str, retry_infra: bool) -> set[tuple[str, str, str, int]]:
    return {
        (r["scenario_id"], r["strategy"], r["model"], r["run"])
        for r in existing_records(model)
        if not (r.get("infrastructure_failure") and retry_infra)
    }
 
 
def make_litellm(model_cfg: dict[str, Any]) -> Any:
    from app.llm import LiteLLM
 
    return LiteLLM(model_cfg["litellm"], temperature=0.0)
 
 
def main(argv: "list[str] | None" = None, make_llm: Callable[[dict[str, Any]], Any] = make_litellm) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", dest="set_name", required=True, choices=["dev", "pilot", "test"])
    ap.add_argument("--model", nargs="+", required=True, help="keys of configs/models.json")
    ap.add_argument("--strategy", nargs="+", default=["react", "adaptive", "local_repair"])
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--lengths", nargs="+", type=int, default=None)
    ap.add_argument("--disruptions", nargs="+", type=int, default=None)
    ap.add_argument("--per-cell", type=int, default=None, help="only the first K scenarios of each cell")
    ap.add_argument("--shard", type=parse_shard, default=None, metavar="I/N")
    ap.add_argument("--worker", default=None, help="your name: results go to files ending __NAME.jsonl")
    ap.add_argument("--uncapped", action="store_true", help="no token cap and no action cap (pilot, FP4 rerun)")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--retry-infra", action="store_true", help="rerun episodes recorded as infrastructure failures")
    ap.add_argument("--allow-commit-change", action="store_true",
                    help="test set only: run from a dirty tree, or resume under a different commit")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
 
    from app.orchestrator import RunSpec, run_episode
    from runner.envfile import load_env
 
    load_env()
    models, caps = load_json("models.json"), load_json("caps.json")
    for name in args.model:
        if name not in models:
            ap.error(f"unknown model {name!r}; choose from {[m for m in models if not m.startswith('_')]}")
    scenarios = scenario_files(args.set_name, args.per_cell, args.lengths, args.disruptions, args.shard)
    commit, dirty = git_state()
 
    if args.set_name == "test" and not args.dry_run and not args.allow_commit_change:
        if dirty:
            ap.error("the test set must be run from a clean checkout (commit your changes first), "
                     "or pass --allow-commit-change if you really mean it")
        for model in args.model:
            others = {r.get("git_commit") for r in existing_records(model)} - {None, commit}
            if others:
                ap.error(f"results for {model} were produced by a different commit ({sorted(others)}); "
                         "mixed code makes the comparison meaningless. Pass --allow-commit-change to override.")
    elif dirty:
        print("note: uncommitted changes in the working tree; records will say git_dirty=true")
 
    total = len(scenarios) * len(args.strategy) * len(args.model) * args.runs
    shard = f" (shard {args.shard[0]}/{args.shard[1]})" if args.shard else ""
    print(f"{len(scenarios)} scenarios{shard} x {len(args.strategy)} strategies x {len(args.model)} models x {args.runs} runs = {total} episodes")
 
    llms: dict[str, Any] = {}
    for run in range(1, args.runs + 1):
        jobs = [(m, sid, s) for m in args.model for sid in scenarios for s in args.strategy]
        if args.shuffle:
            random.Random(f"{args.seed}-{run}").shuffle(jobs)
        done = {m: done_keys(m, args.retry_infra) for m in args.model}
        for model, sid, strategy in jobs:
            if (sid, strategy, model, run) in done[model]:
                continue
            if args.dry_run:
                print(f"would run run={run} {model} {strategy} {sid}")
                continue
            llms.setdefault(model, make_llm(models[model]))
            out_dir = RESULTS / model / "traces" / args.set_name / strategy / f"{sid}__run{run}"
            record: dict[str, Any] = {}
            for attempt in range(MAX_RERUNS + 1):
                shutil.rmtree(out_dir, ignore_errors=True)
                spec = RunSpec(
                    scenario_path=str(scenarios[sid]), strategy=strategy, model=model, run=run,
                    token_cap=token_cap(caps, sid), uncapped=args.uncapped, out_dir=str(out_dir),
                    model_config=models[model],
                )
                record = run_episode(spec, llm=llms[model], response_format=models[model]["response_format"])
                if not record["infrastructure_failure"]:
                    break
            record.update(reruns=attempt, set=args.set_name, git_commit=commit, git_dirty=dirty, worker=args.worker)
            stem = f"{args.set_name}__{strategy}" + (f"__{args.worker}" if args.worker else "")
            path = RESULTS / model / f"{stem}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as fh:  # append one line at a time: a crash loses at most one episode
                fh.write(json.dumps(record) + "\n")
            print(f"run={run} {model} {strategy} {sid}: success={record.get('success')} "
                  f"stop={record['stop_reason']} reruns={attempt}")
    return 0
 
 
if __name__ == "__main__":
    sys.exit(main())