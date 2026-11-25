"""The MCP connection under stress: everything the next step needs to not error."""
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
 
import pytest
 
pytest.importorskip("mcp")
 
from app.orchestrator import RunSpec, _server_env, run_episode
from app.tools import MCPToolClient
from runner.envfile import load_env
from runner.scripted import ScriptedLLM, adaptive_script, first_with, react_script, repair_script, scenario_path
 
D0 = scenario_path("dev", "dev_L10_D0_000.json")
 
 
def client(tmp_path, scenario=D0, **kw):
    return MCPToolClient(sys.executable, ["-m", "errandworld.server", "--scenario", str(scenario),
                                          "--out", str(tmp_path)], env=_server_env(), **kw)
 
 
def test_local_repair_detours_a_closed_road_through_the_real_server(tmp_path):
    path = first_with("dev", "road_closure")
    replies, detour_len = repair_script(path)
    record = run_episode(RunSpec(str(path), "local_repair", "scripted", out_dir=str(tmp_path)), llm=ScriptedLLM(replies))
    assert record["success"] is True and record["stop_reason"] == "goal_reached"
    assert record["divergences"] == 0 and record["revisions"] == 1
    trace = json.loads((tmp_path / "trace.json").read_text())
    revision = trace["m4"]["revisions"][0]
    assert revision["scope"] == "step" and revision["write_cap"] == detour_len + 2 and revision["cut"] == 0
 
 
def test_all_three_loops_run_through_the_real_server(tmp_path):
    for strategy, path, replies in [
        ("react", D0, react_script(D0)),
        ("adaptive", D0, adaptive_script(D0)),
    ]:
        record = run_episode(RunSpec(str(path), strategy, "scripted", out_dir=str(tmp_path / strategy)), llm=ScriptedLLM(replies))
        assert record["success"] and record["divergences"] == 0, strategy
 
 
def test_a_stuck_server_call_times_out_instead_of_hanging(tmp_path):
    with client(tmp_path) as c:
        started = time.time()
        with pytest.raises(TimeoutError):
            c._await(asyncio.sleep(30), timeout=0.3)
        assert time.time() - started < 3
 
 
@pytest.mark.skipif(sys.platform == "win32", reason="uses pgrep and SIGKILL")
def test_a_server_killed_mid_episode_is_an_infrastructure_failure_not_a_hang(tmp_path):
    c = client(tmp_path)
    c.start()
    c.call("status", {})
    pid = int(subprocess.run(["pgrep", "-f", str(tmp_path)], capture_output=True, text=True).stdout.split()[0])
    os.kill(pid, signal.SIGKILL)
    time.sleep(0.5)
    with pytest.raises(Exception):  # an MCP "connection closed" error, raised at once
        c.call("look", {})
    c.close()
 
 
def test_a_server_that_cannot_start_raises_promptly(tmp_path):
    bad = MCPToolClient(sys.executable, ["-m", "errandworld.server", "--scenario", "/nope.json", "--out", str(tmp_path)],
                        env=_server_env())
    started = time.time()
    with pytest.raises(Exception):
        bad.start()
    bad.close()
    assert time.time() - started < 20
 
 
@pytest.mark.skipif(sys.platform == "win32", reason="uses pgrep")
def test_no_server_process_is_left_behind_after_close(tmp_path):
    with client(tmp_path) as c:
        c.call("status", {})
    time.sleep(0.5)
    assert subprocess.run(["pgrep", "-f", str(tmp_path)], capture_output=True, text=True).stdout.strip() == ""
 
 
def test_dotenv_is_loaded_without_overriding_real_environment(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nA_KEY=from_file\nB_KEY='quoted'\nEMPTY=\nC_KEY=file\n")
    monkeypatch.setenv("C_KEY", "real")
    for k in ("A_KEY", "B_KEY", "EMPTY"):
        monkeypatch.delenv(k, raising=False)
    assert sorted(load_env(env)) == ["A_KEY", "B_KEY"]
    assert os.environ["A_KEY"] == "from_file" and os.environ["B_KEY"] == "quoted"
    assert os.environ["C_KEY"] == "real" and "EMPTY" not in os.environ
    for k in ("A_KEY", "B_KEY"):
        os.environ.pop(k, None)
 
 
def test_m5_recovery_lists_every_disruption_and_whether_its_jobs_were_done(tmp_path):
    from runner.scripted import scenario_path
    path = scenario_path("dev", "dev_L10_D1_000.json")
    record = run_episode(RunSpec(str(path), "adaptive", "scripted", out_dir=str(tmp_path)), llm=ScriptedLLM(adaptive_script(path)))
    trace = json.loads((tmp_path / "trace.json").read_text())
    [entry] = trace["m5"]["recovery"]
    assert entry["fired"] is True and entry["affected_jobs"] and entry["recovered"] is True
    assert record["recoverable_disruptions"] == 1 and record["recovered_disruptions"] == 1