"""Local repair's write cap: hidden from the model, enforced by cutting the reply, and counted."""
import json

import pytest

from errandworld.episode import Episode
from errandworld.scenario_io import load_scenario
from metrics.repair_stats import main as repair_stats_main
from metrics.repair_stats import repair_rows
from runner.scripted import ScriptedLLM, first_with, plan_json, repair_script


def rec(cell=(10, 1), strategy="local_repair", revisions=1, capped=1, truncated=0, cut=0, **kw):
    length, disruptions = cell
    return {"scenario_id": f"pilot_L{length}_D{disruptions}_{kw.pop('n', 0):03d}", "strategy": strategy,
            "model": "m", "run": 1, "set": "pilot", "length_level": length, "disruption_level": disruptions,
            "revisions": revisions, "repairs_capped": capped, "repairs_truncated": truncated,
            "steps_cut": cut, **kw}


def test_rows_per_cell_then_overall_from_local_repair_only():
    rows, missing = repair_rows([
        rec((10, 1), n=0, revisions=2, capped=2, truncated=1, cut=3),
        rec((10, 1), n=1, revisions=1, capped=1, truncated=0, cut=0),
        rec((20, 3), n=0, revisions=3, capped=2, truncated=2, cut=2),
        rec((20, 3), n=1, strategy="adaptive", revisions=4, capped=0),
        rec((20, 3), n=2, infrastructure_failure=True, revisions=1, capped=1, truncated=1, cut=1),
    ])
    assert missing == 0
    assert [r["cell"] for r in rows] == ["L10_D1", "L20_D3", "all"]
    l10, l20, overall = rows
    assert (l10["episodes"], l10["repairs"], l10["capped"], l10["truncated"]) == (2, 3, 3, 1)
    assert l10["cut_share"] == pytest.approx(1 / 3) and l10["steps_cut"] == 3 and l10["episodes_cut"] == 1
    assert (l20["episodes"], l20["cut_share"], l20["episodes_cut"]) == (1, 1.0, 1)
    assert (overall["episodes"], overall["capped"], overall["truncated"], overall["steps_cut"]) == (3, 5, 3, 5)


def test_cut_share_is_undefined_without_capped_repairs_and_old_records_are_counted_apart():
    old = rec(n=1)
    del old["repairs_capped"]
    rows, missing = repair_rows([rec(n=0, revisions=1, capped=0), old])
    assert missing == 1
    assert rows[0]["cut_share"] is None and rows[0]["episodes"] == 1


def test_report_reads_every_worker_file_and_prints_each_cell(tmp_path, capsys):
    folder = tmp_path / "m"
    folder.mkdir()
    (folder / "pilot__local_repair__ana.jsonl").write_text(json.dumps(rec(n=0, truncated=1, cut=2)) + "\n")
    (folder / "pilot__local_repair__ben.jsonl").write_text(
        json.dumps(rec(n=0, truncated=1, cut=2)) + "\n" + json.dumps(rec((30, 0), n=0, revisions=0, capped=0)) + "\n"
    )
    assert repair_stats_main([str(folder), "--set", "pilot"]) == 0
    out = capsys.readouterr().out
    assert "1 duplicate episode(s) ignored" in out
    assert "L10_D1" in out and "L30_D0" in out and "undefined" in out


def overshooting_repair(path):
    """Local-repair replies whose repair is one write over the cap: the detour around the closed
    road, then back, forth and back along its last road. The cap (detour + 2) keeps the first
    two of those, so the agent still ends where the detour does, and the final move is cut."""
    replies, detour_len = repair_script(path)
    detour = [s["args"]["location"] for s in json.loads(replies[1])["steps"]]
    ep = Episode(load_scenario(path))
    for s in json.loads(replies[0])["steps"]:
        if not ep.call(s["tool"], s["args"]).ok:
            break
    route = [ep.world.agent_location] + detour
    back, there = route[-2], route[-1]
    patch = [("move", {"location": n}) for n in detour + [back, there, back]]
    return [replies[0], plan_json(patch)], detour_len


def test_a_repair_over_the_cap_is_cut_and_counted_through_the_real_server(tmp_path):
    pytest.importorskip("mcp")
    from app.orchestrator import RunSpec, run_episode

    path = first_with("dev", "road_closure")
    replies, detour_len = overshooting_repair(path)
    record = run_episode(RunSpec(str(path), "local_repair", "scripted", out_dir=str(tmp_path)), llm=ScriptedLLM(replies))

    assert record["success"] is True and record["stop_reason"] == "goal_reached"
    assert record["divergences"] == 0 and record["revisions"] == 1
    assert (record["repairs_capped"], record["repairs_truncated"], record["steps_cut"]) == (1, 1, 1)
    revision = json.loads((tmp_path / "trace.json").read_text())["m4"]["revisions"][0]
    assert revision["write_cap"] == detour_len + 2 and revision["cut"] == 1
