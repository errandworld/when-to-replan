import dataclasses
import json
from pathlib import Path

import pytest

from errandworld.generator import (
    GenerationError,
    GeneratorConfig,
    _sample_jobs,
    balanced_disruption_types,
    cell_disruption_types,
    composition,
    generate_scenario,
    stable_seed,
    trigger_points,
)
from errandworld.progress import ProgressTracker
from errandworld.recipes import CRAFTED_TYPES, RECIPES
from errandworld.scenario_io import load_map, scenario_problems
from errandworld.solver import solve
from errandworld.types import DisruptionType, JobKind
from errandworld.world import World

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def test_map():
    return load_map(ROOT / "data/maps/test_map_1.json")


# ---------------------------------------------------------------- helpers


@pytest.mark.parametrize("total", range(2, 20))
@pytest.mark.parametrize("k", [0, 1, 3])
def test_trigger_points_valid(total, k):
    if total - 1 < k:
        with pytest.raises(GenerationError):
            trigger_points(total, k)
        return
    pts = trigger_points(total, k)
    assert len(pts) == k
    assert pts == sorted(set(pts))
    assert all(1 <= p <= total - 1 for p in pts)


def test_trigger_points_evenly_spaced():
    assert trigger_points(12, 3) == [3, 6, 9]
    assert trigger_points(6, 1) == [3]
    assert trigger_points(4, 3) == [1, 2, 3]


def test_balanced_types():
    for n in (0, 1, 5, 30, 90):
        types = balanced_disruption_types(n, seed=1)
        counts = [types.count(t) for t in DisruptionType]
        assert len(types) == n and max(counts) - min(counts) <= 1
    assert balanced_disruption_types(30, 1) == balanced_disruption_types(30, 1)


@pytest.mark.parametrize("n", [5, 30])
@pytest.mark.parametrize("k", [0, 1, 3])
def test_cell_disruption_types_distinct_and_balanced(n, k):
    per = cell_disruption_types(n, k, seed=3)
    assert len(per) == n and all(len(x) == k for x in per)
    assert all(len(set(x)) == k for x in per)  # distinct within a scenario
    if k:
        flat = [t for x in per for t in x]
        counts = [flat.count(t) for t in DisruptionType]
        assert max(counts) - min(counts) <= 1
    assert per == cell_disruption_types(n, k, seed=3)


def test_stable_seed_is_deterministic_and_distinct():
    assert stable_seed("test", 10, 3, 0) == stable_seed("test", 10, 3, 0)
    assert stable_seed("test", 10, 3, 0) != stable_seed("pilot", 10, 3, 0)


def test_planned_composition_is_exact_on_average():
    cfg = GeneratorConfig()
    assert [cfg.planned_composition(10, i) for i in range(2)] == [(1, 0), (1, 0)]
    assert [cfg.planned_composition(20, i)[0] for i in range(4)] == [1, 2, 1, 2]
    assert [cfg.planned_composition(30, i) for i in range(2)] == [(2, 0), (2, 0)]
    crafts20 = sum(cfg.planned_composition(20, i)[0] for i in range(30))
    assert crafts20 / (30 * 3) == 0.5


def test_band():
    cfg = GeneratorConfig()
    assert cfg.band(10) == (9, 11)
    assert cfg.band(20) == (18, 22)
    assert cfg.band(30) == (27, 33)


# ---------------------------------------------------------------- job sampling rules


@pytest.mark.parametrize("seed", range(100))
def test_sampled_jobs_follow_rules(seed):
    import random

    rng = random.Random(seed)
    n_jobs = rng.choice([2, 3, 4, 6])
    n_craft = rng.randint(0, n_jobs)
    n_dep = rng.randint(0, min(n_craft // 2, 3))
    specs = _sample_jobs(rng, n_jobs, n_craft, n_dep)
    kinds = [k for k, _, _ in specs]
    assert kinds.count("craft") == n_craft
    assert sum(p is not None for _, _, p in specs) == n_dep
    craft_outputs = [t for k, t, _ in specs if k == "craft"]
    deliver = [t for k, t, _ in specs if k == "deliver"]
    craft_inputs = {i for t in craft_outputs for i in RECIPES[t].inputs}
    assert len(set(craft_outputs)) == len(craft_outputs)
    assert len(set(deliver)) == len(deliver)
    assert not set(deliver) & craft_inputs  # no re-delivery loophole
    assert not set(deliver) & CRAFTED_TYPES
    for i, (k, t, p) in enumerate(specs):
        if p is not None:
            assert p < i and RECIPES[specs[p][1]].output in RECIPES[t].inputs


def test_infeasible_composition_raises():
    import random

    with pytest.raises(GenerationError):
        _sample_jobs(random.Random(0), 2, 3, 0)
    with pytest.raises(GenerationError):
        _sample_jobs(random.Random(0), 4, 2, 2)


# ---------------------------------------------------------------- generate_scenario


@pytest.mark.parametrize("L", [10, 20, 30])
@pytest.mark.parametrize("D", [0, 1, 3])
def test_generated_scenario_properties(test_map, L, D):
    types = balanced_disruption_types(D, seed=L + D)
    s = generate_scenario(
        test_map, scenario_id=f"t_L{L}_D{D}", set_name="unit", length_level=L,
        disruption_types=types, seed=stable_seed("unit", L, D), index=1,
    )
    cfg = GeneratorConfig()
    assert scenario_problems(s) == []
    lo, hi = cfg.band(L)
    assert lo <= s.optimal_actions <= hi
    assert s.disruption_level == D == len(s.schedule)
    assert sorted(d.type for d in s.schedule) == sorted(types)  # order may be rearranged
    assert len(s.metadata["reference_replay"]) == D
    assert [d.trigger_subgoal for d in s.schedule] == trigger_points(s.total_subgoals, D)
    assert s.total_subgoals == ProgressTracker(s.jobs).total_subgoals()
    comp = composition(s.jobs)
    assert (comp["craft"], comp["dependent"]) == cfg.planned_composition(L, 1)
    # every needed item exists exactly once; deliver items do not start at their destination
    types_present = [i.type for i in s.items]
    for j in s.jobs:
        needed = [j.target_type] if j.kind is JobKind.DELIVER else [
            t for t in RECIPES[j.target_type].inputs if t not in CRAFTED_TYPES
        ]
        for t in needed:
            assert types_present.count(t) == 1
        if j.kind is JobKind.DELIVER:
            item = next(i for i in s.items if i.type == j.target_type)
            assert item.location != j.destination
    # the recorded optimum is reproducible
    assert solve(World.from_scenario(s), ProgressTracker(s.jobs)).cost == s.optimal_actions


def test_generation_is_deterministic(test_map):
    kw = dict(scenario_id="d", set_name="unit", length_level=20,
              disruption_types=[DisruptionType.STALE_MAP], seed=42, index=0)
    a, b = generate_scenario(test_map, **kw), generate_scenario(test_map, **kw)
    assert a == b and a.metadata["attempt"] == b.metadata["attempt"]


def test_dependent_jobs_can_be_enabled(test_map):
    cfg = dataclasses.replace(GeneratorConfig(), dependent_jobs={10: 0, 20: 0, 30: 1})
    s = generate_scenario(
        test_map, scenario_id="dep", set_name="unit", length_level=30, disruption_types=[],
        seed=7, index=0, config=cfg,
    )
    deps = [j for j in s.jobs if j.depends_on]
    assert len(deps) == 1 and scenario_problems(s) == []


def test_unknown_length_level(test_map):
    with pytest.raises(ValueError):
        generate_scenario(test_map, scenario_id="x", set_name="unit", length_level=40,
                          disruption_types=[], seed=0)


def test_gives_up_with_reasons(test_map):
    cfg = dataclasses.replace(GeneratorConfig(), max_attempts=3, tolerance=0.0,
                              jobs_per_level={10: 4, 20: 3, 30: 4})
    with pytest.raises(GenerationError, match="too_long"):
        generate_scenario(test_map, scenario_id="x", set_name="unit", length_level=10,
                          disruption_types=[], seed=0, config=cfg)


# ---------------------------------------------------------------- committed datasets


@pytest.mark.parametrize("set_name, n", [("dev", 45), ("pilot", 45), ("test", 270)])
def test_committed_dataset_manifests(set_name, n):
    path = ROOT / "data/scenarios" / set_name / "manifest.json"
    if not path.exists():
        pytest.skip("dataset not generated")
    m = json.loads(path.read_text())
    assert m["n_scenarios"] == n
    assert sum(c["n"] for c in m["cells"].values()) == n
    for cell in m["cells"].values():
        assert cell["dependent_jobs"] == 0
        assert "unproven" not in cell["rejections"]
        counts = list(cell["disruption_types"].values())
        if counts:
            assert max(counts) - min(counts) <= 1  # balanced within each cell


def test_sets_are_disjoint():
    keys = {}
    for set_name in ("dev", "pilot", "test"):
        path = ROOT / "data/scenarios" / set_name / "manifest.json"
        if not path.exists():
            pytest.skip("dataset not generated")
        for sid, info in json.loads(path.read_text())["scenarios"].items():
            assert info["content_key"] not in keys, (sid, keys.get(info["content_key"]))
            keys[info["content_key"]] = sid
