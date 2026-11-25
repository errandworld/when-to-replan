"""Is the harness ready for the first real run? Checks everything that can be checked without a model.
 
    python -m runner.preflight                         # offline, about 15 seconds, costs nothing
    python -m runner.preflight --model gemini-flash-lite --live   # also three tiny calls per model
 
Exit code 1 if any check FAILs. WARN means "will run, but fix before the pilot". Nothing here
tunes a prompt or runs a real episode: the loops are driven by a scripted stand-in for the model,
so what is tested is the plumbing (server, client, shadow episode, caps, traces), not the prompts.
"""
from __future__ import annotations
 
import argparse
import contextlib
import importlib.metadata
import importlib.util
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse
 
from runner.envfile import load_env
from runner.run import ROOT, SCENARIOS, load_json
 
Result = "tuple[str, str]"  # (PASS | WARN | FAIL | SKIP, detail)
EXPECTED = {"dev": 45, "pilot": 45, "test": 270}
 
 
def check_python() -> "Result":
    v = sys.version_info
    return ("PASS" if v >= (3, 11) else "FAIL", f"Python {v.major}.{v.minor}.{v.micro} (needs 3.11 or newer)")
 
 
def check_errandworld() -> "Result":
    """`import errandworld` finds the repo's outer errandworld/ folder (no __init__.py) unless the
    package is installed, and the error that follows is confusing. Say it plainly instead."""
    try:
        found = importlib.util.find_spec("errandworld.episode") is not None
    except ImportError:
        found = False
    if not found:
        return "FAIL", "the errandworld package is not installed. Run: pip install -e './errandworld[server]'"
    import errandworld.episode
 
    return "PASS", f"errandworld importable from {Path(errandworld.episode.__file__).parent}"
 
 
def check_mcp() -> "Result":
    try:
        version = importlib.metadata.version("mcp")
        from mcp import ClientSession, StdioServerParameters, types  # noqa: F401
        from mcp.client.stdio import stdio_client  # noqa: F401
        from mcp.server.lowlevel import Server  # noqa: F401
        from mcp.server.stdio import stdio_server  # noqa: F401
    except Exception as exc:
        return "FAIL", f"mcp is not usable ({type(exc).__name__}: {exc}). Run: pip install -r requirements.txt"
    if version.split(".")[0] != "2":
        return "FAIL", f"mcp {version} installed, but the server and client are written for 2.x. Run: pip install 'mcp>=2.3,<3'"
    return "PASS", f"mcp {version}"
 
 
def check_litellm() -> "Result":
    try:
        return "PASS", f"litellm {importlib.metadata.version('litellm')}"
    except importlib.metadata.PackageNotFoundError:
        return "FAIL", "litellm is not installed. Run: pip install -r requirements.txt"
 
 
def check_data() -> "Result":
    counts = {s: len([p for p in (SCENARIOS / s).glob("*.json") if p.stem != "manifest"]) for s in EXPECTED}
    bad = {s: n for s, n in counts.items() if n != EXPECTED[s]}
    return ("FAIL", f"wrong scenario counts {bad} (expected {EXPECTED})") if bad else ("PASS", f"scenarios {counts}")
 
 
def check_configs(models: list[str]) -> "Result":
    try:
        cfg, caps, prices = load_json("models.json"), load_json("caps.json"), load_json("prices.json")
    except Exception as exc:
        return "FAIL", f"a config file does not parse: {exc}"
    missing = [m for m in models if m not in cfg]
    if missing:
        return "FAIL", f"models not in configs/models.json: {missing}"
    todo = []
    if any(cfg[m].get("pinned_version") is None for m in models):
        todo.append("pinned_version unset")
    if prices.get("date") is None or any(v is None for m in models for v in prices["models"].get(m, {}).values()):
        todo.append("prices unset")
    if any(v is not None for k, v in caps.items() if not k.startswith("_")):
        todo.append("caps are set, so runs are capped unless --uncapped")
    return ("WARN", "; ".join(todo)) if todo else ("PASS", "models, caps and prices are filled in")
 
 
def check_keys(models: list[str]) -> "Result":
    cfg, problems = load_json("models.json"), []
    for m in models:
        name = cfg[m]["litellm"]
        if name.startswith("gemini/") and not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            problems.append(f"{m}: GEMINI_API_KEY not set")
        elif name.startswith("nvidia_nim/") and not os.environ.get("NVIDIA_NIM_API_KEY"):
            problems.append(f"{m}: NVIDIA_NIM_API_KEY not set")
        elif name.startswith("ollama"):
            base = urlparse(os.environ.get("OLLAMA_API_BASE") or "http://localhost:11434")
            try:
                socket.create_connection((base.hostname, base.port or 11434), timeout=2).close()
            except OSError:
                problems.append(f"{m}: nothing is listening at {base.hostname}:{base.port or 11434} (is `ollama serve` running?)")
    return ("FAIL", "; ".join(problems)) if problems else ("PASS", f"credentials found for {', '.join(models)}")
 
 
def check_roundtrip() -> "Result":
    from runner.scripted import scenario_path
    from app.orchestrator import _server_env
    from app.tools import MCPToolClient
    from errandworld.tools import TOOL_SPECS
 
    out = Path(tempfile.mkdtemp(prefix="preflight_"))
    path = scenario_path("dev", "dev_L10_D0_000.json")
    started = time.time()
    with MCPToolClient(sys.executable, ["-m", "errandworld.server", "--scenario", str(path), "--out", str(out)],
                       env=_server_env()) as client:
        startup = time.time() - started
        defs = {d["function"]["name"]: d["function"] for d in client.definitions()}
        if set(defs) != set(TOOL_SPECS):
            return "FAIL", f"server offers {sorted(defs)}, expected {sorted(TOOL_SPECS)}"
        for name, spec in TOOL_SPECS.items():
            if defs[name]["parameters"] != spec.parameters or defs[name]["description"] != spec.description:
                return "FAIL", f"tool {name!r} reaches the client with a different schema or description than tools.py defines"
        for name, args in [("status", {}), ("look", {}), ("read_map", {}), ("pick", {"item": "nothing_1"}),
                           ("move", {"location": "nowhere"}), ("drop", {"item": "nothing_1"}),
                           ("use", {"station": "none_1", "items": []}), ("pick", {})]:
            r = client.call(name, args)
            if not isinstance(r.text, str) or not r.text:
                return "FAIL", f"{name} returned no text for the model to read"
            if r.ok and args.get("item") == "nothing_1":
                return "FAIL", f"{name} of an item that does not exist should fail"
    time.sleep(0.5)
    trace = json.loads((out / "trace.json").read_text())
    if trace["summary"]["end_reason"] != "terminated":
        return "FAIL", f"server did not close its books on disconnect (end_reason={trace['summary']['end_reason']})"
    if sys.platform != "win32" and subprocess.run(["pgrep", "-f", str(out)], capture_output=True, text=True).stdout.strip():
        return "FAIL", "a server process was left running after close"
    return "PASS", f"7 tools, schemas identical to tools.py, errors return normally, clean shutdown, started in {startup:.1f}s"
 
 
def check_failure_modes() -> "Result":
    from app.orchestrator import _server_env
    from app.tools import MCPToolClient
 
    out = tempfile.mkdtemp(prefix="preflight_")
    bad = MCPToolClient(sys.executable, ["-m", "errandworld.server", "--scenario", "/nope.json", "--out", out], env=_server_env(),
                        errlog=open(os.devnull, "w"))  # the crash is the point; keep its traceback out of the report
    started = time.time()
    try:
        bad.start()
        return "FAIL", "a server that cannot start was reported as started"
    except Exception:
        pass
    finally:
        bad.close()
    took = time.time() - started
    return ("PASS", f"a server that cannot start raises in {took:.1f}s instead of hanging") if took < 30 else ("FAIL", f"took {took:.0f}s to give up")
 
 
def check_loops() -> "Result":
    from app.orchestrator import RunSpec, run_episode
    from runner.scripted import ScriptedLLM, adaptive_script, first_with, react_script, repair_script, scenario_path
 
    d0 = scenario_path("dev", "dev_L10_D0_000.json")
    road = first_with("dev", "road_closure")
    repair_replies, detour = repair_script(road)
    runs = [("react", d0, react_script(d0)), ("adaptive", scenario_path("dev", "dev_L10_D1_000.json"), adaptive_script(scenario_path("dev", "dev_L10_D1_000.json"))),
            ("local_repair", road, repair_replies)]
    notes = []
    for strategy, path, replies in runs:
        record = run_episode(RunSpec(str(path), strategy, "scripted", out_dir=tempfile.mkdtemp(prefix="preflight_")), llm=ScriptedLLM(replies))
        if not (record["success"] and record["divergences"] == 0 and not record["infrastructure_failure"]):
            return "FAIL", f"{strategy}: {record['stop_reason']}, success={record['success']}, divergences={record['divergences']}, error={record['error']}"
        notes.append(f"{strategy}({record['revisions']} rev)")
    return "PASS", "scripted model solved a scenario with each loop against the real server: " + ", ".join(notes)
 
 
def check_runner(models: list[str]) -> "Result":
    from runner import run as runner
 
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        runner.main(["--set", "pilot", "--model", models[0], "--per-cell", "1", "--dry-run"])
    n = buffer.getvalue().count("would run")
    return ("PASS", f"runner dry run lists {n} episodes") if n == 27 else ("FAIL", f"dry run listed {n} episodes, expected 27")
 
 
def check_live(models: list[str]) -> "Result":
    """Three tiny calls per model: it answers, it makes a tool call, it follows the plan schema."""
    from app.llm import LiteLLM
    from app.planner import _response_format
    from errandworld.tools import TOOL_SPECS
 
    defs = [{"type": "function", "function": {"name": s.name, "description": s.description, "parameters": s.parameters}} for s in TOOL_SPECS.values()]
    cfg, notes = load_json("models.json"), []
    for m in models:
        llm = LiteLLM(cfg[m]["litellm"])
        try:
            hello = llm.complete([{"role": "user", "content": "Reply with one word: ok"}])
            tool = llm.complete([{"role": "user", "content": "You are in a town. Use the look tool now."}], tools=defs)
            plan = llm.complete([{"role": "user", "content": "Write a plan with one step that calls the look tool."}],
                                response_format=_response_format(cfg[m]["response_format"], defs))
        except Exception as exc:
            return "FAIL", f"{m}: {type(exc).__name__}: {str(exc)[:200]}"
        problems = []
        if not tool.tool_calls or tool.tool_calls[0].name != "look":
            problems.append("did not call a tool when asked (FP3 11.2: reliable function calling)")
        try:
            if "steps" not in json.loads(plan.text):
                problems.append("plan reply has no steps")
        except json.JSONDecodeError:
            problems.append("plan reply is not JSON")
        if problems:
            return "FAIL", f"{m}: " + "; ".join(problems)
        notes.append(f"{m} -> version {hello.model_version!r}; paste into configs/models.json pinned_version")
    return "PASS", " | ".join(notes)
 
 
def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", nargs="+", default=None, help="keys of configs/models.json (default: all)")
    ap.add_argument("--live", action="store_true", help="also make three small real API calls per model")
    args = ap.parse_args(argv)
    loaded = load_env()
    models = args.model or [m for m in load_json("models.json") if not m.startswith("_")]
 
    checks: list[tuple[str, Callable[[], "Result"]]] = [
        ("python", check_python), ("errandworld package", check_errandworld), ("mcp", check_mcp), ("litellm", check_litellm), ("scenario data", check_data),
        ("configs", lambda: check_configs(models)), ("credentials", lambda: check_keys(models)),
        ("mcp round trip", check_roundtrip), ("server failure modes", check_failure_modes),
        ("three loops, real server", check_loops), ("runner", lambda: check_runner(models)),
    ]
    if args.live:
        checks.append(("live model calls", lambda: check_live(models)))
    failed = False
    print(f".env: {len(loaded)} variable(s) loaded\n")
    needs_install = {"mcp round trip", "server failure modes", "three loops, real server", "runner", "live model calls"}
    blocked = False
    for name, fn in checks:
        if blocked and name in needs_install:
            print(f"SKIP  {name:26} needs the package and mcp installed")
            continue
        try:
            status, detail = fn()
        except Exception as exc:  # a check that crashes is a failed check
            status, detail = "FAIL", f"{type(exc).__name__}: {exc}"
        failed |= status == "FAIL"
        blocked |= status == "FAIL" and name in ("errandworld package", "mcp")
        print(f"{status:5} {name:26} {detail}")
    print("\n" + ("NOT READY: fix the FAIL lines first." if failed else "READY for the next step."))
    return 1 if failed else 0
 
 
if __name__ == "__main__":
    sys.exit(main())