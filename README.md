# when-to-replan

**ErrandWorld** is a controlled testbed for studying when LLM agents benefit from explicit, revisable plans. It independently varies task length and plan-invalidating disruptions to separate the challenges of long tasks from those of changing environments. The project compares ReAct, adaptive replanning, and local repair across three models under matched token and action budgets, using a deterministic simulator and ground truth computed by a classical search planner.

## Layout

```
errandworld/     the simulator, solver, generator, scenarios, and the MCP server (Part A)
  errandworld/     package: world, tools, solver, disruptions, episode, wire.py, server.py
  data/            maps, scenarios (dev 45, pilot 45, test 270), validation reports
  scripts/         generate and validate datasets, play a scenario by hand
  tests/           simulator tests
app/             the agent side (Part B): loops, planner, checker, executor, LiteLLM, MCP client
  orchestrator.py  runs one episode: server + shadow + budget + loop + trace
  prompts/         environment.md (the rules sheet), react.md, plan_*.md
runner/          the experiment runner and the token-cap derivation (Part C)
metrics/         failure categories, bootstrap, cost, per-cell summary (Part D)
configs/         models.json, caps.json, prices.json
results/         per-model JSONL and traces (pilot results are committed)
tests/           agent-side, runner and end-to-end tests
docs/            FP reports, CHANGES_SINCE_FP2.md, prompt_revisions.md
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp example.env .env        # then fill in the keys
python -m runner.preflight # is everything ready? fix any FAIL line before running anything
pytest                     # agent side + end to end (about 30 s); `cd errandworld && pytest` for the simulator
```

## Run

```bash
python -m runner.preflight --model gemini-flash-lite --live   # three tiny real calls: answers, calls a tool, follows the plan schema
python -m app.llm gemini/gemini-3.5-flash-lite          # one cheap call: shows the version string the API returns
python -m runner.run --set pilot --model gemini-flash-lite --per-cell 2 --uncapped --dry-run
python -m runner.run --set pilot --model gemini-flash-lite --per-cell 2 --uncapped
python -m runner.derive_caps                            # 1.5 x median ReAct tokens -> configs/caps.json
python -m metrics.summarize results/gemini-flash-lite   # success, delta with CI, cost per success
python -m metrics.repair_stats results/gemini-flash-lite  # how often local repair's hidden cap cut a repair
```

Strategy (`--strategy react adaptive local_repair`), model, set and run count are all command-line choices.
An interrupted run resumes: finished (scenario, strategy, model, run) tuples are skipped.

## Splitting the work across the team

Everyone must be on the same git commit, with the same `configs/` and prompts. Each person picks a slice
and a worker name; results go to their own files, so git merges cleanly:

```bash
# Four people share the pilot by scenario (all three strategies of a scenario stay together)
python -m runner.run --set pilot --model gemini-flash-lite nemotron-120b --uncapped --shard 1/4 --worker ana
python -m runner.run --set pilot --model gemini-flash-lite nemotron-120b --uncapped --shard 2/4 --worker ben
python -m runner.run --set pilot --model gemini-flash-lite nemotron-120b --uncapped --shard 3/4 --worker caro
python -m runner.run --set pilot --model gemini-flash-lite nemotron-120b --uncapped --shard 4/4 --worker dee

# or by cell: only the 4-disruption cells, only length 10 and 20
python -m runner.run --set pilot --model qwen-27b --disruptions 4 --lengths 10 20 --worker dee
```

Then commit your `results/<model>/*.jsonl` files, pull everyone else's, and run
`python -m runner.derive_caps` and `python -m metrics.summarize results/* --set pilot`.
An episode two people both ran is counted once. The **test set** refuses to run from a dirty tree
or on top of results from a different commit (override with `--allow-commit-change`).
