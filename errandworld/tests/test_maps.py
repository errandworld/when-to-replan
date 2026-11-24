import pytest

from conftest import make_tiny_map
from errandworld.maps import (
    adjacency,
    bridges,
    describe,
    diameter,
    distances_from,
    generate_map,
    is_connected,
    map_invariant_problems,
)
from errandworld.recipes import STATION_TYPES
from errandworld.scenario_io import map_problems
from errandworld.types import Road


@pytest.mark.parametrize("seed", range(50))
@pytest.mark.parametrize("deg", [2.0, 2.4, 3.0])
def test_generated_maps_satisfy_invariants(seed, deg):
    m = generate_map("m", seed, avg_degree=deg)
    assert map_problems(m) == []
    assert map_invariant_problems(m) == []
    assert len(m.locations) == 15
    assert len(m.roads) == round(deg * 15 / 2)
    assert sorted(s.type for s in m.stations) == sorted(STATION_TYPES * 2)


@pytest.mark.parametrize("seed", range(30))
def test_closing_any_single_road_keeps_map_connected(seed):
    import dataclasses

    m = generate_map("m", seed)
    for i, r in enumerate(m.roads):
        roads = list(m.roads)
        roads[i] = Road(r.a, r.b, open=False)
        assert is_connected(dataclasses.replace(m, roads=tuple(roads)))


def test_generation_is_deterministic():
    assert generate_map("m", 7) == generate_map("m", 7)
    assert generate_map("m", 7) != generate_map("m", 8)


def test_sparser_maps_are_longer():
    sparse = [diameter(generate_map("m", s, avg_degree=2.0)) for s in range(20)]
    dense = [diameter(generate_map("m", s, avg_degree=3.0)) for s in range(20)]
    assert sum(sparse) > sum(dense)


def test_invalid_parameters():
    with pytest.raises(ValueError):
        generate_map("m", 0, avg_degree=1.5)
    with pytest.raises(ValueError):
        generate_map("m", 0, n_locations=4)  # 6 stations cannot fit one per location
    with pytest.raises(ValueError):
        generate_map("m", 0, n_locations=100)


def test_graph_utilities_on_tiny_map():
    m = make_tiny_map()  # home-market-workshop, home-park-office-market
    assert adjacency(m)["market"] == ["home", "office", "workshop"]
    assert distances_from(m, "workshop") == {
        "workshop": 0, "market": 1, "home": 2, "office": 2, "park": 3,
    }
    assert is_connected(m)
    assert diameter(m) == 3
    assert bridges(m) == [("market", "workshop")]  # workshop is a leaf
    probs = map_invariant_problems(m)
    assert any("bridges" in p for p in probs)
    assert any("'oven'" in p for p in probs)


def test_bridges_respect_closed_roads():
    import dataclasses

    m = make_tiny_map()
    closed = tuple(Road(r.a, r.b, open=not (r.a, r.b) == ("home", "park")) for r in m.roads)
    m2 = dataclasses.replace(m, roads=closed)
    # With home-park closed the open roads form a tree, so every remaining road is a bridge.
    assert bridges(m2) == [
        ("home", "market"), ("market", "office"), ("market", "workshop"), ("office", "park"),
    ]


def test_describe():
    d = describe(generate_map("m", 3, avg_degree=2.4))
    assert d["locations"] == 15 and d["roads"] == 18 and d["bridges"] == 0
    assert d["avg_degree"] == 2.4 and d["diameter"] >= 3


def test_committed_map_files_match_their_seeds():
    """data/maps/*.json must be exactly what scripts/generate_maps.py produces."""
    import importlib.util
    from pathlib import Path

    from errandworld.scenario_io import load_map

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("generate_maps", root / "scripts/generate_maps.py")
    gm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gm)
    for map_id in gm.MAP_SEEDS:
        assert load_map(root / "data/maps" / f"{map_id}.json") == gm.build(map_id), map_id
