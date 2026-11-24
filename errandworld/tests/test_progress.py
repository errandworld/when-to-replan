"""Progress tracking: subgoals, crediting rules, latching, transfers, and invariants."""

import random

import pytest

from conftest import make_tiny_scenario
from errandworld.progress import (
    DELIVER,
    PICK,
    USE,
    ProgressTracker,
    Subgoal,
    count_subgoals,
    describe_job,
    subgoals_for,
)
from errandworld.tools import TOOL_SPECS, dispatch
from errandworld.types import Item, Job, JobKind
from errandworld.world import World


class Run:
    """Drive a world through the tool layer and feed results to a tracker."""

    def __init__(self, scenario):
        self.world = World.from_scenario(scenario)
        self.tracker = ProgressTracker(scenario.jobs)
        self.events = []

    def __call__(self, name, **args):
        r = dispatch(self.world, name, args, jobs=self.tracker.jobs_view())
        ev = self.tracker.observe(r.tool, r.ok, r.data)
        self.events += ev
        return r, ev

    @property
    def count(self):
        return self.tracker.subgoals_completed()


@pytest.fixture
def run(tiny_scenario):
    return Run(tiny_scenario)


def lamp_job_steps(run):
    run("move", location="market")
    run("pick", item="lamp_base_1")
    run("move", location="workshop")
    run("pick", item="bulb_1")
    run("use", station="repair_bench_1", items=["lamp_base_1", "bulb_1"])


# ---------------------------------------------------------------- definitions


def test_subgoals_for_each_job_kind():
    deliver = Job("d", JobKind.DELIVER, "battery", "office")
    craft = Job("c", JobKind.CRAFT, "lamp", "home")
    assert [(s.kind, s.item_type) for s in subgoals_for(deliver)] == [
        (PICK, "battery"), (DELIVER, "battery"),
    ]
    assert [(s.kind, s.item_type) for s in subgoals_for(craft)] == [
        (PICK, "lamp_base"), (PICK, "bulb"), (USE, "lamp"), (DELIVER, "lamp"),
    ]
    assert count_subgoals([deliver, craft]) == 6


def test_job_descriptions_are_pinned():
    a = Job("job_1", JobKind.CRAFT, "radio", "market")
    b = Job("job_2", JobKind.CRAFT, "tuned_radio", "office", ("job_1",))
    assert describe_job(Job("j", JobKind.DELIVER, "battery", "office")) == (
        "Deliver a battery to office."
    )
    assert describe_job(a) == (
        "Make a radio at a repair_bench from radio_shell + battery, then deliver it to market."
    )
    assert describe_job(b) == (
        "Make a tuned_radio at a repair_bench from radio + antenna, then deliver it to office. "
        "Use the radio delivered for job_1."
    )


def test_task_description_and_jobs_view(tiny_scenario):
    t = ProgressTracker(tiny_scenario.jobs)
    assert t.task_description() == (
        "Complete all of these jobs:\n"
        "- job_1: Deliver a battery to office.\n"
        "- job_2: Make a lamp at a repair_bench from lamp_base + bulb, then deliver it to home."
    )
    assert t.jobs_view() == [
        {"id": "job_1", "description": "Deliver a battery to office.", "done": False},
        {
            "id": "job_2",
            "description": "Make a lamp at a repair_bench from lamp_base + bulb, then deliver it to home.",
            "done": False,
        },
    ]


def test_total_matches_scenario_field(tiny_scenario):
    assert ProgressTracker(tiny_scenario.jobs).total_subgoals() == tiny_scenario.total_subgoals == 6


# ---------------------------------------------------------------- happy path


def test_full_trajectory_completes_every_subgoal(run):
    run("move", location="market")
    run("pick", item="battery_1")
    lamp_job_steps(run)
    run("move", location="market")
    run("move", location="office")
    _, ev = run("drop", item="battery_1")
    assert run.tracker.job_done("job_1") and [e.subgoal.kind for e in ev] == [DELIVER]
    run("move", location="market")
    run("move", location="home")
    run("drop", item="lamp_1")
    assert run.tracker.all_done()
    assert run.count == 6 and run.tracker.progress_rate() == 1.0
    assert [e.count for e in run.events] == [1, 2, 3, 4, 5, 6]
    assert not any(e.inferred for e in run.events)
    assert [e.subgoal.label for e in run.events] == [
        "job_1:pick:battery", "job_2:pick:lamp_base", "job_2:pick:bulb", "job_2:use:lamp",
        "job_1:deliver:battery", "job_2:deliver:lamp",
    ]


def test_status_shows_job_progress(run):
    run("move", location="market")
    run("pick", item="battery_1")
    run("move", location="office")
    run("drop", item="battery_1")
    r, _ = run("status")
    assert "- job_1 [done]: Deliver a battery to office." in r.text
    assert "- job_2 [pending]:" in r.text


# ---------------------------------------------------------------- no inflation


def test_failed_and_read_calls_never_count(run):
    run("pick", item="battery_1")  # not here
    run("look")
    run("read_map")
    run("status")
    run("move", location="workshop")  # not adjacent
    assert run.count == 0 and run.events == []


def test_distractor_pick_does_not_count(run):
    run("move", location="market")
    run("move", location="home")
    run("move", location="park")
    _, ev = run("pick", item="ink_1")
    assert ev == [] and run.count == 0


def test_repicking_same_type_does_not_count(run):
    run("move", location="market")
    run("pick", item="battery_1")
    run("drop", item="battery_1")  # market is not job_1's destination
    _, ev = run("pick", item="battery_1")
    assert ev == [] and run.count == 1


def test_input_pick_after_use_does_not_count():
    extra = (Item("lamp_base_2", "lamp_base", "workshop"),)
    run = Run(make_tiny_scenario(items=make_tiny_scenario().items + extra))
    lamp_job_steps(run)
    assert run.count == 3
    _, ev = run("pick", item="lamp_base_2")
    assert ev == [] and run.count == 3


def test_drop_at_wrong_place_does_not_complete(run):
    run("move", location="market")
    run("pick", item="battery_1")
    run("move", location="home")
    _, ev = run("drop", item="battery_1")  # job_2 destination, but wrong type
    assert ev == [] and not run.tracker.job_done("job_1")


# ---------------------------------------------------------------- latching (D4)


def test_completion_is_latched(run):
    run("move", location="market")
    run("pick", item="battery_1")
    run("move", location="office")
    run("drop", item="battery_1")
    assert run.tracker.job_done("job_1") and run.count == 2
    _, ev = run("pick", item="battery_1")
    assert ev == []
    run("move", location="market")
    run("drop", item="battery_1")
    assert run.tracker.job_done("job_1") and run.count == 2


# ---------------------------------------------------------------- transfers & dependencies


def two_battery_jobs():
    return make_tiny_scenario(
        items=(
            Item("battery_1", "battery", "market"),
            Item("battery_2", "battery", "market"),
        ),
        jobs=(
            Job("job_a", JobKind.DELIVER, "battery", "office"),
            Job("job_b", JobKind.DELIVER, "battery", "home"),
        ),
        schedule=(),
        disruption_level=0,
        total_subgoals=4,
    )


def test_delivery_transfers_pick_credit_between_same_type_jobs():
    run = Run(two_battery_jobs())
    run("move", location="market")
    _, ev = run("pick", item="battery_1")
    assert ev[0].subgoal == Subgoal("job_a", PICK, "battery")  # first pending job in order
    run("move", location="home")
    _, ev = run("drop", item="battery_1")  # completes job_b instead
    assert run.tracker.job_done("job_b")
    assert [e.subgoal.label for e in ev] == ["job_b:deliver:battery"]  # pick was transferred
    assert run.count == 2
    assert run.tracker.job_subgoals("job_a") == {
        "job_a:pick:battery": False, "job_a:deliver:battery": False,
    }
    run("move", location="market")
    _, ev = run("pick", item="battery_2")
    assert ev[0].subgoal == Subgoal("job_a", PICK, "battery")
    assert run.count == 3


def dependent_radio_jobs():
    return make_tiny_scenario(
        items=(
            Item("radio_shell_1", "radio_shell", "workshop"),
            Item("battery_1", "battery", "workshop"),
            Item("antenna_1", "antenna", "workshop"),
        ),
        jobs=(
            Job("job_1", JobKind.CRAFT, "radio", "market"),
            Job("job_2", JobKind.CRAFT, "tuned_radio", "office", ("job_1",)),
        ),
        schedule=(),
        disruption_level=0,
        total_subgoals=8,
    )


def craft_radio(run):
    run("move", location="market")
    run("move", location="workshop")
    run("pick", item="radio_shell_1")
    run("pick", item="battery_1")
    run("use", station="repair_bench_1", items=["radio_shell_1", "battery_1"])


def test_dependent_job_consumes_delivered_output():
    run = Run(dependent_radio_jobs())
    craft_radio(run)
    run("pick", item="antenna_1")
    run("move", location="market")
    run("drop", item="radio_1")
    assert run.tracker.job_done("job_1") and run.count == 5
    _, ev = run("pick", item="radio_1")  # consume job_1's delivered radio
    assert ev[0].subgoal == Subgoal("job_2", PICK, "radio")
    run("move", location="workshop")
    run("use", station="repair_bench_1", items=["radio_1", "antenna_1"])
    run("move", location="market")
    run("move", location="office")
    run("drop", item="tuned_radio_1")
    assert run.tracker.all_done()
    assert run.tracker.job_done("job_1")  # latched although its radio was consumed
    assert run.count == 8
    assert not any(e.inferred for e in run.events)


def test_skipping_producer_delivery_is_flagged_inferred():
    """Using the crafted radio directly (never delivering it) credits job_2's radio pick by
    inference, and job_1 can no longer be completed."""
    run = Run(dependent_radio_jobs())
    craft_radio(run)
    run("pick", item="antenna_1")
    _, ev = run("use", station="repair_bench_1", items=["radio_1", "antenna_1"])
    assert [(e.subgoal.label, e.inferred) for e in ev] == [
        ("job_2:pick:radio", True), ("job_2:use:tuned_radio", False),
    ]
    assert not run.tracker.job_done("job_1")


# ---------------------------------------------------------------- invariants


@pytest.mark.parametrize(
    "scenario_factory", [make_tiny_scenario, two_battery_jobs, dependent_radio_jobs]
)
def test_random_play_is_monotone_bounded_and_consistent(scenario_factory):
    rng = random.Random(0)
    names = list(TOOL_SPECS)
    for _ in range(150):
        run = Run(scenario_factory())
        w = run.world
        last = 0
        for _ in range(60):
            name = rng.choice(names)
            here = w.agent_location
            if name == "move":
                args = {"location": rng.choice(w.neighbors(here))}
            elif name == "pick":
                cands = w.items_at(here) or ["nothing"]
                args = {"item": rng.choice(cands)}
            elif name == "drop":
                args = {"item": rng.choice(w.inventory or ["nothing"])}
            elif name == "use":
                st = w.stations_at(here) or ["nothing"]
                inv = list(w.inventory)
                rng.shuffle(inv)
                args = {"station": rng.choice(st), "items": inv[:2] or ["nothing"]}
            else:
                args = {}
            _, ev = run(name, **args)
            c = run.count
            assert last <= c <= run.tracker.total_subgoals()
            assert [e.count for e in ev] == list(range(last + 1, c + 1))
            last = c
            done_jobs = [j for j in run.tracker.jobs if run.tracker.job_done(j.id)]
            for j in done_jobs:
                assert all(run.tracker.job_subgoals(j.id).values())
        assert run.tracker.all_done() == (run.count == run.tracker.total_subgoals())


# ---------------------------------------------------------------- copy / serialize


def test_clone_is_independent(run):
    run("move", location="market")
    run("pick", item="battery_1")
    c = run.tracker.clone()
    key = run.tracker.state_key()
    c.observe("drop", True, {"type": "battery", "location": "office", "item": "battery_1"})
    assert c.job_done("job_1") and not run.tracker.job_done("job_1")
    assert run.tracker.state_key() == key != c.state_key()


def test_snapshot(run):
    run("move", location="market")
    run("pick", item="battery_1")
    snap = run.tracker.snapshot()
    assert snap["subgoals_completed"] == 1 and snap["total_subgoals"] == 6
    assert snap["jobs"][0] == {
        "id": "job_1",
        "done": False,
        "subgoals": {"job_1:pick:battery": True, "job_1:deliver:battery": False},
    }
