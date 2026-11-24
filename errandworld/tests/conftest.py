"""Shared fixtures: a tiny hand-built scenario reused by later test modules."""

from __future__ import annotations

import pytest

from errandworld.types import (
    DisruptionSpec,
    DisruptionType,
    Item,
    Job,
    JobKind,
    Location,
    MapSpec,
    Road,
    Scenario,
    Station,
)


def make_tiny_map() -> MapSpec:
    """A 5-location map:

        home — market — workshop
          |        |
        park — office
    """
    locs = ("home", "market", "workshop", "park", "office")
    return MapSpec(
        id="tiny_map",
        seed=0,
        locations=tuple(Location(l, l.title()) for l in locs),
        roads=(
            Road("home", "market"),
            Road("market", "workshop"),
            Road("home", "park"),
            Road("park", "office"),
            Road("market", "office"),
        ),
        stations=(
            Station("repair_bench_1", "repair_bench", "workshop"),
            Station("repair_bench_2", "repair_bench", "park"),
        ),
    )


def make_tiny_scenario(**overrides) -> Scenario:
    """Two jobs: deliver a battery to the office; craft a lamp and deliver it home.
    One scheduled road closure after the 2nd subgoal."""
    fields = dict(
        id="tiny_L10_D1_000",
        seed=123,
        set_name="unit",
        map=make_tiny_map(),
        start_location="home",
        items=(
            Item("battery_1", "battery", "market"),
            Item("lamp_base_1", "lamp_base", "market"),
            Item("bulb_1", "bulb", "workshop"),
            Item("ink_1", "ink", "park"),  # distractor
        ),
        jobs=(
            Job("job_1", JobKind.DELIVER, "battery", "office"),
            Job("job_2", JobKind.CRAFT, "lamp", "home"),
        ),
        schedule=(DisruptionSpec(0, DisruptionType.ROAD_CLOSURE, 2),),
        capacity=3,
        length_level=10,
        disruption_level=1,
        total_subgoals=6,
    )
    fields.update(overrides)
    return Scenario(**fields)


@pytest.fixture
def tiny_scenario() -> Scenario:
    return make_tiny_scenario()
