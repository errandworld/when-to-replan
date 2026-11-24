import dataclasses

import pytest

from errandworld.types import DisruptionType, JobKind, Road, road_key


def test_road_endpoints_are_normalized():
    r = Road("zoo", "apple")
    assert (r.a, r.b) == ("apple", "zoo")
    assert r == Road("apple", "zoo")
    assert r.key == road_key("zoo", "apple") == ("apple", "zoo")


def test_road_other_endpoint():
    r = Road("a", "b")
    assert r.other("a") == "b"
    assert r.other("b") == "a"
    with pytest.raises(ValueError):
        r.other("c")


def test_road_rejects_self_loop():
    with pytest.raises(ValueError):
        Road("a", "a")


def test_specs_are_immutable(tiny_scenario):
    with pytest.raises(dataclasses.FrozenInstanceError):
        tiny_scenario.capacity = 5  # type: ignore[misc]


def test_scenarios_are_hashable_and_ignore_metadata(tiny_scenario):
    other = dataclasses.replace(tiny_scenario, metadata={"note": "x"})
    assert other == tiny_scenario
    assert hash(other) == hash(tiny_scenario)


def test_enums_serialize_as_strings():
    assert JobKind.CRAFT == "craft"
    assert DisruptionType("stale_map") is DisruptionType.STALE_MAP
