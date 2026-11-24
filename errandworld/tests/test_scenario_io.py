import dataclasses
import json

import pytest

from conftest import make_tiny_map, make_tiny_scenario
from errandworld.scenario_io import (
    SCHEMA_VERSION,
    ScenarioError,
    fingerprint,
    load_map,
    load_scenario,
    save_map,
    save_scenario,
    scenario_from_dict,
    scenario_problems,
    scenario_to_dict,
)
from errandworld.types import DisruptionSpec, DisruptionType, Item, Job, JobKind


def test_tiny_scenario_is_valid(tiny_scenario):
    assert scenario_problems(tiny_scenario) == []


def test_scenario_round_trip_through_file(tmp_path, tiny_scenario):
    s = dataclasses.replace(tiny_scenario, metadata={"attempts": 3})
    path = tmp_path / "s.json"
    save_scenario(s, path)
    loaded = load_scenario(path)
    assert loaded == s
    assert loaded.metadata == {"attempts": 3}
    assert json.loads(path.read_text())["schema_version"] == SCHEMA_VERSION


def test_map_round_trip_through_file(tmp_path):
    m = make_tiny_map()
    save_map(m, tmp_path / "m.json")
    assert load_map(tmp_path / "m.json") == m


def test_fingerprint_is_stable_and_content_sensitive(tiny_scenario):
    fp = fingerprint(tiny_scenario)
    assert fp == fingerprint(make_tiny_scenario())
    assert fp == fingerprint(dataclasses.replace(tiny_scenario, metadata={"x": 1}))
    assert fp != fingerprint(dataclasses.replace(tiny_scenario, seed=999))


def test_rejects_wrong_schema_version(tiny_scenario):
    d = scenario_to_dict(tiny_scenario)
    d["schema_version"] = 99
    with pytest.raises(ScenarioError, match="schema_version"):
        scenario_from_dict(d)


def test_rejects_malformed_dict(tiny_scenario):
    d = scenario_to_dict(tiny_scenario)
    del d["jobs"]
    with pytest.raises(ScenarioError, match="malformed"):
        scenario_from_dict(d)


def _problems(**overrides):
    return " | ".join(scenario_problems(make_tiny_scenario(**overrides)))


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"start_location": "mars"}, "start_location"),
        ({"capacity": 1}, "capacity"),
        (
            {"items": (Item("battery_1", "battery", "mars"),)},
            "unknown location 'mars'",
        ),
        (
            {"items": (Item("x", "battery", "home"), Item("x", "bulb", "home"))},
            "duplicate item id 'x'",
        ),
        ({"jobs": ()}, "no jobs"),
        (
            {"jobs": (Job("j", JobKind.DELIVER, "battery", "mars"),)},
            "destination 'mars'",
        ),
        ({"jobs": (Job("j", JobKind.DELIVER, "lamp", "home"),)}, "is crafted; use CRAFT"),
        ({"jobs": (Job("j", JobKind.DELIVER, "frame", "home"),)}, "no item of type 'frame'"),
        ({"jobs": (Job("j", JobKind.CRAFT, "spaceship", "home"),)}, "no recipe"),
        ({"jobs": (Job("j", JobKind.CRAFT, "bread", "home"),)}, "no 'oven' station"),
        (
            {"jobs": (Job("j", JobKind.CRAFT, "radio", "home"),)},
            "no item of input type 'radio_shell'",
        ),
        (
            {"jobs": (Job("j", JobKind.DELIVER, "battery", "home", ("ghost",)),)},
            "unknown job 'ghost'",
        ),
        (
            {"schedule": (DisruptionSpec(0, DisruptionType.MOVED_ITEM, 0),)},
            "must be >= 1",
        ),
        (
            {"schedule": (DisruptionSpec(0, DisruptionType.MOVED_ITEM, 6),)},
            "must be < total_subgoals",
        ),
        (
            {
                "schedule": (
                    DisruptionSpec(0, DisruptionType.MOVED_ITEM, 4),
                    DisruptionSpec(1, DisruptionType.ROAD_CLOSURE, 2),
                )
            },
            "not ordered",
        ),
        ({"disruption_level": 3}, "disruption_level 3"),
    ],
)
def test_validation_catches_problems(overrides, expected):
    assert expected in _problems(**overrides)


def test_dependent_craft_job_requires_producer():
    # tuned_radio consumes a crafted radio; without a depends_on job it is invalid.
    items = (
        Item("radio_shell_1", "radio_shell", "home"),
        Item("battery_1", "battery", "home"),
        Item("antenna_1", "antenna", "home"),
    )
    bad = make_tiny_scenario(
        items=items, jobs=(Job("j2", JobKind.CRAFT, "tuned_radio", "office"),)
    )
    assert any("no depends_on job produces it" in p for p in scenario_problems(bad))

    good = make_tiny_scenario(
        items=items,
        jobs=(
            Job("j1", JobKind.CRAFT, "radio", "market"),
            Job("j2", JobKind.CRAFT, "tuned_radio", "office", ("j1",)),
        ),
    )
    assert scenario_problems(good) == []


def test_dependency_must_consume_output_and_be_acyclic():
    items = (
        Item("radio_shell_1", "radio_shell", "home"),
        Item("battery_1", "battery", "home"),
    )
    unrelated = make_tiny_scenario(
        items=items,
        jobs=(
            Job("j1", JobKind.DELIVER, "battery", "office"),
            Job("j2", JobKind.CRAFT, "radio", "home", ("j1",)),
        ),
    )
    # radio does consume battery, but battery is a base item, not j1's crafted output.
    assert any("does not consume its crafted output" in p for p in scenario_problems(unrelated))

    cyclic = make_tiny_scenario(
        items=items,
        jobs=(
            Job("j1", JobKind.CRAFT, "radio", "home", ("j2",)),
            Job("j2", JobKind.CRAFT, "radio", "home", ("j1",)),
        ),
    )
    assert any("cycle" in p for p in scenario_problems(cyclic))


def test_save_refuses_invalid_scenario(tmp_path):
    with pytest.raises(ScenarioError):
        save_scenario(make_tiny_scenario(start_location="mars"), tmp_path / "bad.json")
    assert not (tmp_path / "bad.json").exists()
