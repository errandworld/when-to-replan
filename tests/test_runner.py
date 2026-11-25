"""Runner mechanics (FP3 section 3) with a model that gives up at once, so no API is touched."""
import json
 
import pytest
 
pytest.importorskip("mcp")
 
from runner import run as runner
from runner.derive_caps import main as derive_caps
from tests.fakes import FakeLLM
 
 
class QuitsImmediately(FakeLLM):
    temperature = 0.0
 
    def __init__(self, *_):
        super().__init__(["I am done."] * 50)
 
 
@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(runner, "load_json", lambda name: (
        {"fake": {"litellm": "fake/fake", "response_format": "schema"}} if name == "models.json" else {}))
    return tmp_path
 
 
def run_main(*extra):
    return runner.main(["--set", "dev", "--model", "fake", "--strategy", "react", "--per-cell", "1", *extra],
                       make_llm=lambda cfg: QuitsImmediately())
 
 
def lines(sandbox):
    path = sandbox / "results" / "fake" / "dev__react.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines()]
 
 
def test_one_jsonl_per_condition_with_the_four_tuple_on_every_record(sandbox):
    run_main()
    records = lines(sandbox)
    assert len(records) == 9                       # one scenario in each of the nine cells
    assert {(r["length_level"], r["disruption_level"]) for r in records} == {(l, d) for l in (10, 20, 30) for d in (0, 1, 3)}
    for r in records:
        assert (r["strategy"], r["model"], r["run"]) == ("react", "fake", 1)
        assert r["scenario_id"].startswith("dev_") and r["success"] is False and r["infrastructure_failure"] is False
    assert (sandbox / "results" / "fake" / "traces" / "dev" / "react").is_dir()
 
 
def test_restart_skips_finished_episodes_and_a_new_run_number_runs_everything_again(sandbox):
    run_main()
    run_main()                                     # same four-tuples: nothing to do
    assert len(lines(sandbox)) == 9
    run_main("--runs", "2")                        # run 1 is done, run 2 is not
    assert sorted({r["run"] for r in lines(sandbox)}) == [1, 2] and len(lines(sandbox)) == 18
 
 
def test_dry_run_touches_nothing(sandbox, capsys):
    run_main("--dry-run")
    assert "would run" in capsys.readouterr().out
    assert not (sandbox / "results").exists()
 
 
def test_token_caps_come_from_configs_by_cell_or_length():
    assert runner.token_cap({"L10": 5, "L10_D3": 9}, "dev_L10_D3_000") == 9
    assert runner.token_cap({"L10": 5}, "dev_L10_D0_000") == 5
    assert runner.token_cap({}, "dev_L20_D1_002") is None
 
 
def test_filters_by_length_and_disruption_count_and_per_cell():
    both = runner.scenario_files("dev", lengths=[10], disruptions=[0, 3])
    assert both and {runner.parse_id(s) for s in both} == {(10, 0), (10, 3)}
    one_each = runner.scenario_files("dev", per_cell=1, disruptions=[3])
    assert {runner.parse_id(s) for s in one_each} == {(10, 3), (20, 3), (30, 3)} and len(one_each) == 3
 
 
def test_shards_partition_the_work_with_no_overlap_and_nothing_lost():
    everything = set(runner.scenario_files("dev"))
    shards = [set(runner.scenario_files("dev", shard=(i, 3))) for i in (1, 2, 3)]
    assert set().union(*shards) == everything
    assert sum(len(s) for s in shards) == len(everything)
    assert runner.parse_shard("2/3") == (2, 3)
    for bad in ("0/3", "4/3", "x", "2"):
        with pytest.raises(Exception):
            runner.parse_shard(bad)
 
 
def test_each_worker_writes_its_own_file_and_resume_sees_everyones_work(sandbox):
    runner.main(["--set", "dev", "--model", "fake", "--strategy", "react", "--per-cell", "1", "--shard", "1/2", "--worker", "ana"],
                make_llm=lambda cfg: QuitsImmediately())
    runner.main(["--set", "dev", "--model", "fake", "--strategy", "react", "--per-cell", "1", "--shard", "2/2", "--worker", "ben"],
                make_llm=lambda cfg: QuitsImmediately())
    folder = sandbox / "results" / "fake"
    assert (folder / "dev__react__ana.jsonl").exists() and (folder / "dev__react__ben.jsonl").exists()
    ids = [json.loads(l)["scenario_id"] for f in folder.glob("*.jsonl") for l in f.read_text().splitlines()]
    assert len(ids) == 9 and len(set(ids)) == 9           # the two workers did different scenarios
    run_main()                                             # a third person running everything finds nothing left
    assert sum(len(f.read_text().splitlines()) for f in folder.glob("*.jsonl")) == 9
    record = json.loads((folder / "dev__react__ana.jsonl").read_text().splitlines()[0])
    assert record["worker"] == "ana" and "git_commit" in record
 
 
def test_the_test_set_refuses_a_dirty_tree_and_mixed_commits(sandbox, monkeypatch):
    monkeypatch.setattr(runner, "git_state", lambda: ("abc123", True))
    with pytest.raises(SystemExit):
        runner.main(["--set", "test", "--model", "fake", "--per-cell", "1"], make_llm=lambda c: QuitsImmediately())
    monkeypatch.setattr(runner, "git_state", lambda: ("abc123", False))
    folder = sandbox / "results" / "fake"
    folder.mkdir(parents=True)
    (folder / "test__react.jsonl").write_text(json.dumps(
        {"scenario_id": "test_L10_D0_000", "strategy": "react", "model": "fake", "run": 1, "git_commit": "old999"}) + "\n")
    with pytest.raises(SystemExit):
        runner.main(["--set", "test", "--model", "fake", "--per-cell", "1"], make_llm=lambda c: QuitsImmediately())
 
 
def test_summaries_and_caps_read_all_workers_filter_by_set_and_count_duplicates_once(sandbox, monkeypatch, capsys):
    from metrics.summarize import load_records
    folder = sandbox / "results" / "fake"
    folder.mkdir(parents=True)
    row = {"scenario_id": "pilot_L10_D0_000", "strategy": "react", "model": "fake", "run": 1, "set": "pilot",
           "length_level": 10, "disruption_level": 0, "success": True, "input_tokens": 100, "cached_input_tokens": 0,
           "output_tokens": 50, "infrastructure_failure": False}
    (folder / "pilot__react__ana.jsonl").write_text(json.dumps(row) + "\n")
    (folder / "pilot__react__ben.jsonl").write_text(json.dumps(row) + "\n" + json.dumps({**row, "set": "dev"}) + "\n")
    records, duplicates = load_records([folder], "pilot")
    assert len(records) == 1 and duplicates == 1
    monkeypatch.setattr("runner.derive_caps.RESULTS", sandbox / "results")
    monkeypatch.setattr("runner.derive_caps.ROOT", sandbox)
    (sandbox / "configs").mkdir()
    derive_caps([])
    assert json.loads((sandbox / "configs" / "caps.json").read_text())["L10"] == 225   # 1.5 x 150, once