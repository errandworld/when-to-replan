"""Human playthrough CLI: command parsing and scripted sessions."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import make_tiny_scenario
from errandworld.types import DisruptionSpec, DisruptionType as DT

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("play", ROOT / "scripts/play.py")
play = importlib.util.module_from_spec(spec)
spec.loader.exec_module(play)


@pytest.mark.parametrize(
    "line, expected",
    [
        ("move market", ("move", {"location": "market"})),
        ("  MOVE market  ", ("move", {"location": "market"})),
        ("pick bulb_1", ("pick", {"item": "bulb_1"})),
        ("drop lamp_1", ("drop", {"item": "lamp_1"})),
        ("use repair_bench_1 lamp_base_1 bulb_1",
         ("use", {"station": "repair_bench_1", "items": ["lamp_base_1", "bulb_1"]})),
        ("look", ("look", {})),
        ("status", ("status", {})),
        ("read_map", ("read_map", {})),
        ('move {"location": "park"}', ("move", {"location": "park"})),
    ],
)
def test_parse_command(line, expected):
    assert play.parse_command(line) == expected


@pytest.mark.parametrize(
    "line, fragment",
    [
        ("", "empty"),
        ("teleport office", "unknown command"),
        ("move", "usage: move"),
        ("move a b", "usage: move"),
        ("pick", "usage: pick"),
        ("use repair_bench_1", "usage: use"),
        ("look around", "takes no arguments"),
        ("move {bad json", "invalid JSON"),
        ('move {"location": "x"', "invalid JSON"),
    ],
)
def test_parse_errors(line, fragment):
    with pytest.raises(play.ParseError, match=fragment):
        play.parse_command(line)


SOLUTION = """
# deliver the battery, craft and deliver the lamp
move market
pick battery_1
pick lamp_base_1
move workshop
pick bulb_1
use repair_bench_1 lamp_base_1 bulb_1
move market
move office
drop battery_1
move market
move home
drop lamp_1
"""


def run(scenario, script):
    out = []
    ep = play.run_session(scenario, script.splitlines(), out=out.append)
    return ep, "\n".join(out)


def test_scripted_solution_succeeds():
    s = make_tiny_scenario(optimal_actions=12, schedule=(), disruption_level=0)
    ep, text = run(s, SOLUTION)
    assert ep.is_success()
    assert "Complete all of these jobs:" in text
    assert "Episode over: success  (SUCCESS)" in text
    assert "Write actions used: 12" in text


def test_input_errors_are_not_sent_and_help_works():
    s = make_tiny_scenario(optimal_actions=12, schedule=(), disruption_level=0)
    ep, text = run(s, "help\nfly away\nmove\nquit\n")
    assert "Commands:" in text
    assert text.count("(input error, not sent:") == 2
    assert ep.write_count == 0 and ep.end_reason == "quit"


def test_disruptions_revealed_only_at_the_end():
    s = make_tiny_scenario(optimal_actions=12, reference_actions=14,
                           schedule=(DisruptionSpec(0, DT.ROAD_CLOSURE, 2),), disruption_level=1)
    ep, text = run(s, "move market\npick battery_1\npick lamp_base_1\nlook\nquit\n")
    before_report = text.split("=" * 60)[0]
    assert "road_closure" not in before_report
    assert "Disruptions:\n  - road_closure: road:" in text
    assert "with disruptions (reference): 14" in text


def test_input_running_out_ends_episode():
    s = make_tiny_scenario(optimal_actions=12, schedule=(), disruption_level=0)
    ep, _ = run(s, "look\n")
    assert ep.end_reason == "quit"


def test_cli_end_to_end(tmp_path):
    sid = "pilot_L10_D0_000"
    if not (ROOT / "data/scenarios/pilot" / f"{sid}.json").exists():
        pytest.skip("dataset not generated")
    trace = tmp_path / "t.json"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/play.py"), sid, "--trace", str(trace)],
        input="status\nread_map\nquit\n", capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Complete all of these jobs:" in proc.stdout and "Locations:" in proc.stdout
    assert trace.exists()
    listing = subprocess.run([sys.executable, str(ROOT / "scripts/play.py"), "--list", "pilot"],
                             capture_output=True, text=True, timeout=60)
    assert len(listing.stdout.split()) == 45
