"""Play an ErrandWorld scenario by hand, through the same text interface an LLM agent gets.

    python scripts/play.py dev_L10_D1_000                 # by scenario id (searched in data/scenarios)
    python scripts/play.py data/scenarios/dev/dev_L20_D3_002.json
    python scripts/play.py dev_L10_D0_000 --trace my_run.json
    python scripts/play.py --list dev                     # list scenario ids in a set

Commands (one per line):
    move <location>            pick <item>            drop <item>
    use <station> <item> [<item> ...]
    look                       status                 read_map
    help                       quit
A tool name followed by a JSON object also works:  move {"location": "ashford"}

You see only what the agent sees: the task description and each tool's result text. Disruptions are
never announced. When the episode ends (or you quit) the harness view is shown: outcome, actions
used vs. the optimum and the cap, and what the disruptions were.

Please note anything confusing in docs/playthrough_notes.md (FP2 section 4.4 asks for ~10 human
playthroughs to check that task descriptions and observations are clear).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from errandworld.episode import RUNNING, Episode  # noqa: E402
from errandworld.scenario_io import load_scenario  # noqa: E402
from errandworld.tools import TOOL_SPECS  # noqa: E402
from errandworld.types import Scenario  # noqa: E402

SCEN_DIR = ROOT / "data" / "scenarios"

HELP = """Commands:
  move <location>          pick <item>          drop <item>
  use <station> <item> [<item> ...]
  look                     status               read_map
  help                     quit
Write actions (move, pick, drop, use) count toward the action limit; look/status/read_map do not."""


class ParseError(ValueError):
    pass


def parse_command(line: str) -> tuple[str, dict]:
    """Turn a typed line into (tool, args). Raises ParseError with a friendly message."""
    line = line.strip()
    if not line:
        raise ParseError("empty command (type 'help')")
    name, _, rest = line.partition(" ")
    name = name.lower()
    rest = rest.strip()
    if name not in TOOL_SPECS:
        raise ParseError(f"unknown command {name!r} (type 'help')")
    if rest.startswith("{"):
        try:
            args = json.loads(rest)
        except json.JSONDecodeError as e:
            raise ParseError(f"invalid JSON arguments: {e}") from None
        if not isinstance(args, dict):
            raise ParseError("JSON arguments must be an object")
        return name, args
    words = rest.split()
    if name in ("look", "status", "read_map"):
        if words:
            raise ParseError(f"{name} takes no arguments")
        return name, {}
    if name == "move":
        if len(words) != 1:
            raise ParseError("usage: move <location>")
        return name, {"location": words[0]}
    if name in ("pick", "drop"):
        if len(words) != 1:
            raise ParseError(f"usage: {name} <item>")
        return name, {"item": words[0]}
    if name == "use":
        if len(words) < 2:
            raise ParseError("usage: use <station> <item> [<item> ...]")
        return name, {"station": words[0], "items": words[1:]}
    raise ParseError(f"unhandled command {name!r}")  # pragma: no cover


def find_scenario(ref: str) -> Path:
    p = Path(ref)
    if p.suffix == ".json" and p.exists():
        return p
    matches = sorted(SCEN_DIR.glob(f"*/{ref}.json"))
    if not matches:
        raise SystemExit(f"no scenario {ref!r} (try --list dev)")
    return matches[0]


def harness_report(ep: Episode) -> str:
    sm = ep.summary()
    lines = [
        "",
        "=" * 60,
        f"Episode over: {sm['end_reason']}  ({'SUCCESS' if sm['success'] else 'not completed'})",
        f"Write actions used: {sm['write_calls']}   (failed: {sm['invalid_write_calls']}; "
        f"read calls: {sm['read_calls']})",
        f"Optimal without disruptions: {sm['optimal_actions']}   "
        f"with disruptions (reference): {sm['reference_actions']}   cap: {sm['action_cap']}",
        f"Jobs done: {', '.join(sm['jobs_done']) or 'none'}   "
        f"subgoals: {sm['subgoals_completed']}/{sm['total_subgoals']}",
    ]
    if ep.disruption_events:
        lines.append("Disruptions:")
        for e in ep.disruption_events:
            if e.fired:
                when = f"after call #{e.step + 1} (subgoal {e.fired_at_subgoal})"
                lines.append(f"  - {e.type.value}: {e.target} {e.detail}  {when}"
                             f"{'  [fallback]' if e.fallback else ''}")
            else:
                lines.append(f"  - {e.type.value}: not fired ({e.reason})")
    lines.append("=" * 60)
    return "\n".join(lines)


def run_session(
    scenario: Scenario,
    lines: Iterable[str],
    out: Callable[[str], None] = print,
    prompt: Callable[[], None] | None = None,
) -> Episode:
    """Drive one episode from an iterable of command lines (stdin or a script)."""
    ep = Episode(scenario)
    out(f"Scenario {scenario.id}\n")
    out(ep.task_description())
    out("\nType 'help' for commands.\n")
    if prompt:
        prompt()
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            if prompt:
                prompt()
            continue
        if line.lower() in ("quit", "exit"):
            ep.end("quit")
            break
        if line.lower() == "help":
            out(HELP)
        else:
            try:
                tool, args = parse_command(line)
            except ParseError as e:
                out(f"(input error, not sent: {e})")
            else:
                out(ep.call(tool, args).text)
        if ep.status != RUNNING:
            break
        if prompt:
            prompt()
    else:
        ep.end("quit")  # input ran out
    out(harness_report(ep))
    return ep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenario", nargs="?", help="scenario id or path to a scenario .json")
    ap.add_argument("--list", metavar="SET", help="list scenario ids in a set (dev, pilot, test)")
    ap.add_argument("--trace", metavar="PATH", help="save the episode trace as JSON")
    a = ap.parse_args()

    if a.list:
        for p in sorted((SCEN_DIR / a.list).glob(f"{a.list}_L*_D*_*.json")):
            print(p.stem)
        return 0
    if not a.scenario:
        ap.error("give a scenario id/path, or --list SET")

    scenario = load_scenario(find_scenario(a.scenario))
    interactive = sys.stdin.isatty()
    ep = run_session(
        scenario,
        sys.stdin,
        prompt=(lambda: print("> ", end="", flush=True)) if interactive else None,
    )
    if a.trace:
        ep.save_trace(a.trace)
        print(f"trace saved to {a.trace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
