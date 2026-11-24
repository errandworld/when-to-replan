# When Is Replanning Worth It? — ErrandWorld harness

CS6983 Final Project, Group 8.

This repo contains **ErrandWorld**, a deterministic text-based errand simulator, and (in later phases)
the agent harness comparing ReAct, adaptive replanning, and local repair under controlled task
length and environmental disruption.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the ErrandWorld design.

## Setup

Requires Python 3.11+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Play a scenario by hand

```bash
python scripts/play.py --list dev                 # scenario ids in a set
python scripts/play.py dev_L20_D1_003             # interactive; type 'help' for commands
python scripts/play.py dev_L20_D1_003 --trace runs/playthroughs/me.json
```

You see exactly what an agent sees; disruptions are revealed only when the episode ends. Record
findings in `docs/playthrough_notes.md` (use dev scenarios only).

## Data

```bash
python scripts/generate_maps.py          # regenerate data/maps/*.json (2 dev + 5 test maps)
python scripts/generate_maps.py --check  # verify committed maps match their seeds
python scripts/generate_datasets.py --set all --workers 6   # 45 dev, 45 pilot, 270 test scenarios
python scripts/generate_datasets.py --verify --set all      # re-check scenario files and manifests
python scripts/validate_datasets.py --set all --workers 6  # solver, replay, and oracle checks
```

## Tests

`pytest` runs the suite (~40 s). The solver is checked against exhaustive brute-force search on
random micro-worlds; `ERRANDWORLD_FULL_SOLVER_TESTS=1 pytest tests/test_solver.py` runs the full
500-seed sweep (~4 min).

## Status

- [x] Step 1 — repo skeleton
- [x] Step 2 — data model (`types.py`), recipe book (`recipes.py`), JSON I/O + validation
      (`scenario_io.py`)
- [x] Step 3 — live world state, transition rules, error codes, map view (`world.py`)
- [x] Step 4 — tool specs, argument validation, dispatch, rendered texts (`tools.py`)
- [x] Step 5 — subgoals, crediting rules, latched completion, job texts (`progress.py`)
- [x] Step 6 — A* solver with proven bounds (`solver.py`), timing benchmark
      (`scripts/bench_solver.py`). Top length lowered to 30 (D6) so every optimum is provable.
- [x] Step 7 — ring-with-chords maps (`maps.py`), fixed maps in `data/maps/`
- [x] Step 8 — scenario generator (`generator.py`), datasets in `data/scenarios/{dev,pilot,test}`
- [x] Step 9 — disruption injector and schedule replay (`disruptions.py`); datasets regenerated
      so every disruption fires strictly on the reference trajectory
- [x] Step 10 — episode wrapper: action cap, ending, M2/M3 logging, traces (`episode.py`)
- [x] Step 11 — oracle agent (`oracle.py`) and dataset validation
      (`scripts/validate_datasets.py`): 360/360 scenarios pass
- [x] Step 12 — human playthrough CLI (`scripts/play.py`), notes in `docs/playthrough_notes.md`

**Phase 1 (ErrandWorld) is complete.** Next: Phase 2, the agent harness (MCP server, the three
strategy loops, runner, LLM-call logging).
