"""Oracle agent and dataset validation."""

import glob
import importlib.util
import json
from pathlib import Path

import pytest

from conftest import make_tiny_scenario
from errandworld.episode import default_action_cap
from errandworld.oracle import run_oracle
from errandworld.scenario_io import load_scenario, scenario_from_dict, scenario_to_dict
from errandworld.types import DisruptionSpec, DisruptionType as DT

ROOT = Path(__file__).resolve().parents[1]


def load_validator():
    spec = importlib.util.spec_from_file_location(
        "validate_datasets", ROOT / "scripts/validate_datasets.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_oracle_succeeds_optimally_without_disruptions():
    s = make_tiny_scenario(optimal_actions=12, schedule=(), disruption_level=0)
    ep = run_oracle(s)
    assert ep.is_success() and ep.write_count == 12 and ep.oracle_replans == 0


def test_oracle_replans_after_disruption():
    s = make_tiny_scenario(optimal_actions=12,
                           schedule=(DisruptionSpec(0, DT.ROAD_CLOSURE, 2),), disruption_level=1)
    ep = run_oracle(s)
    assert ep.is_success() and ep.oracle_replans >= 1
    assert ep.write_count > 12  # the closure cost a detour


def test_oracle_gives_up_cleanly_when_unsolvable():
    """The only road to the workshop (and its bulb) is closed from the start."""
    import dataclasses

    from conftest import make_tiny_map
    from errandworld.types import Road

    m = make_tiny_map()
    roads = tuple(Road(r.a, r.b, open=(r.a, r.b) != ("market", "workshop")) for r in m.roads)
    s = make_tiny_scenario(optimal_actions=12, schedule=(), disruption_level=0,
                           map=dataclasses.replace(m, roads=roads))
    ep = run_oracle(s)
    assert not ep.is_success() and ep.end_reason == "oracle_no_plan" and ep.write_count == 0


def test_reference_actions_round_trip_and_cap():
    s = make_tiny_scenario(optimal_actions=12, reference_actions=17)
    assert scenario_from_dict(scenario_to_dict(s)) == s
    assert default_action_cap(s) == 2 * 17 + 5
    assert default_action_cap(make_tiny_scenario(optimal_actions=12)) == 29


@pytest.mark.parametrize("path", sorted(glob.glob(str(ROOT / "data/scenarios/pilot/pilot_L*_D*_00[01].json"))))
def test_validate_one_passes_on_pilot_samples(path):
    r = load_validator().validate_one(path)
    assert r["ok"], r["problems"]
    s = load_scenario(path)
    assert r["oracle_writes"] == s.reference_actions
    assert r["cap_headroom"] == s.reference_actions + 5


def test_validator_detects_a_corrupted_scenario(tmp_path):
    src = sorted(glob.glob(str(ROOT / "data/scenarios/pilot/pilot_L20_D1_*.json")))[0]
    d = json.loads(Path(src).read_text())
    d["optimal_actions"] += 1
    d["reference_actions"] += 1
    bad = tmp_path / "pilot_L20_D1_999.json"
    bad.write_text(json.dumps(d))
    r = load_validator().validate_one(str(bad))
    assert not r["ok"]
    assert any("!= recorded" in p for p in r["problems"])
    assert any("reference_actions" in p for p in r["problems"])


@pytest.mark.parametrize("set_name", ["dev", "pilot", "test"])
def test_committed_validation_reports_all_pass(set_name):
    path = ROOT / "data/validation" / f"{set_name}_report.json"
    if not path.exists():
        pytest.skip("validation not run")
    rep = json.loads(path.read_text())
    assert rep["n_passed"] == rep["n_scenarios"] and rep["failures"] == []
