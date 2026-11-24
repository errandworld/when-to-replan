# ErrandWorld Architecture

ErrandWorld is a deterministic, text-based errand simulator. It is the environment for the study
*When Is Replanning Worth It?* (CS6983 Group 8), which compares ReAct, adaptive replanning, and
local repair while varying task length and environmental disruption independently.

The system has two halves built on one shared core:

- **Offline pipeline** — runs once; builds maps and scenarios and proves each scenario is solvable
  and well-formed.
- **Runtime** — runs once per episode; plays one scenario against one agent, injects disruptions,
  and records everything.

## Overview

```
                         ┌──────────────────────── OFFLINE (scripts/) ─────────────────────────┐
                         │                                                                      │
 fixed seeds ──► maps.py ──► data/maps/*.json ──► generator.py ──► data/scenarios/{set}/*.json   │
                         │   (15 locations,       (jobs, items,      + manifest.json            │
                         │    roads, stations)     schedule)              │                     │
                         │                            │ solve()            ▼                     │
                         │                            └──────► solver ◄── validate_datasets.py   │
                         │                                     ▲          (replay + oracle)      │
                         └─────────────────────────────────────┼─────────────────────────────────┘
                                                               │  same code
                         ┌──────────────────────── RUNTIME (one episode) ───────────────────────┐
                         │                                     │                                 │
   scenario.json ──────► │  Episode  (episode.py)              │                                 │
                         │  ┌──────────────────────────────────┼────────────────────────────┐   │
   agent / MCP server    │  │  tools.dispatch()                │                            │   │
   ── call(tool,args) ──►│  │     │ validate args              │                            │   │
   ◄── ToolResult(text) ─│  │     ▼                            │                            │   │
                         │  │  World (world.py)          ProgressTracker   Injector         │   │
                         │  │   true state ───write ok──►  (progress.py) ─► (disruptions.py) │   │
                         │  │   map_view                   subgoal events    │  picks target │   │
                         │  │     ▲                                          │  via solver   │   │
                         │  │     └──────────── mutates world ───────────────┘               │   │
                         │  └───────────────────────────────────────────────────────────────┘   │
                         │     harness-only: events() · is_success() · final_state()             │
                         └───────────────────────────────────────────────────────────────────────┘
```

## Layers

Dependencies point strictly downward: the solver and injector never import the episode or the
tool layer.

| Layer | Module(s) | Responsibility | Depends on |
|---|---|---|---|
| Data model | `types.py`, `recipes.py`, `scenario_io.py` | Plain dataclasses (Location, Road, Item, Station, Recipe, Job, DisruptionSpec, Scenario) and versioned JSON round-trip. No logic. | — |
| World state | `world.py` | Single source of truth: map graph, item positions, station status, agent location and inventory, plus a separate `map_view` (the agent's possibly stale picture). Transition rules, error codes, harness-only mutators, `clone()`. | data model |
| Tool interface | `tools.py` | The 7 agent tools: schemas, read/write flags, argument validation, and `ToolResult(ok, error, data, text)`. The only translation from world state to what the agent reads. | world |
| Task progress | `progress.py` | Turns successful writes into subgoals (relevant pick, station use, delivery) and latched job completion. Defines "how far along" for triggers and metrics. | data model |
| Solver | `solver.py` | A\* over write actions: optimal cost, optimal plan, solvability. Operates on a cloned world; never mutates the live one. | world, progress |
| Disruptions | `disruptions.py` | Watches subgoal counts; at trigger points chooses a target whose disruption raises the optimal remaining cost while keeping the task solvable, then applies one of four disruption types. | world, progress, solver |
| Episode | `episode.py` | Wires world, tools, progress, and injector for one run. Exposes `call()` to the agent; exposes the event log, success check, and final state to the harness. Enforces the action cap. | all above |
| Generation & validation | `maps.py`, `generator.py`, `oracle.py`, `scripts/` | Builds maps and scenarios, keeps only those within ±10% of the target optimal length, replays disruption schedules, runs the oracle agent. | episode, solver |

## Maps

Each map (`maps.py`) is a **ring with chords**: 15 locations in a random cycle plus random extra
roads up to an average degree of 2.4 (18 roads). A ring has **no bridges**, so closing any single
road keeps the map connected and every road is a legal closure target; the sparse ring gives long
trips (diameter 5–7). Each station type has 2 stations at distinct locations, at most one station
per location. Location names are neutral place names (Ashford, Birchwood, ...). The seven maps
(2 dev, 5 test) are fixed by seeds in `scripts/generate_maps.py` and committed under `data/maps/`.

## Tools

| Tool | Type | Effect |
|---|---|---|
| `move(location)` | write | Travel along an open road to an adjacent location |
| `pick(item)` | write | Pick up an item at the current location (subject to capacity) |
| `drop(item)` | write | Put down a carried item; completes a delivery at its destination |
| `use(station, items)` | write | Run a station recipe on carried items, producing an output item |
| `look()` | read | Items, stations (online/offline), and roads (open/closed) here — always true state |
| `status()` | read | Own location, inventory, and job progress (done/pending) |
| `read_map()` | read | Roads, stations, item locations from `map_view` — may be stale |

Every tool returns an `ok` flag; failures carry a fixed error code (`not_adjacent`,
`road_closed`, `item_not_here`, `inventory_full`, `not_carrying`, `not_at_station`,
`station_offline`, `missing_items`, `invalid_recipe`, `unknown_id`), plus two raised by the tool
layer itself (`unknown_tool`, `invalid_arguments`). Error texts always begin `ERROR <code>:`.
Arguments are validated strictly (no missing, extra, or mistyped arguments), and a failed call
never changes the world. Write tools count toward the action cap; read tools are limited only by
the token cap.

## Tasks and progress

- A **task** is a set of jobs: about 2, 3, or 4 jobs targeting about 10, 20, or 30 optimal write
  actions (D6); travel distance makes up the rest of the length.
- **Deliver job:** bring an item of a given type to a destination — 2 subgoals (pick, drop).
- **Craft job:** collect two inputs, use a station, deliver the output — 4 subgoals
  (pick, pick, use, drop).
- **Dependent jobs:** a craft job may consume the item another job delivered (supported, but
  disabled in the generated datasets; see D7).
- Job completion is **latched**: once delivered, a job stays complete even if its item is later
  picked up again (needed for dependent jobs).
- Subgoals are credited **by item type**, to the first pending job (in job order) that still
  needs that subgoal. Redundant picks earn nothing. When a station use or delivery implies a
  subgoal the job lacks, the credit is transferred from another pending job holding it, so the
  count never double-counts; if no donor exists it is granted and flagged `inferred`. The subgoal
  count is therefore monotone and bounded by the total (see `progress.py`).

## Disruptions

Disruptions fire at scheduled **subgoal counts** (evenly spaced: `round(i·N/(k+1))` for
`i = 1..k`), immediately after the write action that reaches the count, so the agent encounters the
change on its next action.

| Type | World change | `map_view` updated? |
|---|---|---|
| Road closure | An open non-bridge road closes (chosen by the target rule below) | Yes |
| Moved item | A needed item on the ground moves to another location | Yes |
| Station offline | A needed station goes offline (an alternative of the same type exists) | Yes |
| Stale map entry | A needed item moves, but the map still shows the old location | **No** |

**Target rule** (`disruptions.py`): a candidate target is valid if the disruption (1) raises the
optimal remaining cost, so it breaks every optimal plan, (2) keeps the task solvable, and (3)
affects at least one pending job, meaning that job's own optimal cost rises. Without (3) a
disruption could raise the joint optimum by spoiling a shared route without hitting any job's
resource, leaving its recovery outcome undefined. All three are checked with the exact solver from
the true current state.

Candidates are tried in a seeded random order (scenario seed + disruption index) and the first
valid one is used. That is a uniform choice among valid targets, usually after about 2 re-solves
(median 3 ms per disruption, max ~2 s). Candidate sets:
- road closure: open non-bridge roads;
- moved item / stale map: (needed item on the ground, any other location);
- station offline: online stations of a type a pending craft job needs.

If nothing is valid, the injector falls back to the first candidate that keeps the task solvable
(`fallback=True`); if none exists, it records `fired=False, reason="no_valid_target"`. It never
fires after all jobs are done (`task_complete`), and `finalize()` records unreached triggers as
`not_reached`. Every event logs its target, before/after optimal costs, affected jobs, and how
many candidates were evaluated (M3).

**Schedule replay at generation time** (FP2 §4.4): a scenario is kept only if, along the reference
trajectory (follow an optimal plan, re-plan optimally after each disruption), every disruption
fires strictly. If the planned order of types fails, other orders of the same types over the same
trigger points are tried before resampling placement; 55 of 240 disrupted scenarios use a
reordered schedule. Each 3-disruption scenario gets three *distinct* types, cycling through the
four possible triples so that types stay balanced within each cell. Duplicates can be infeasible:
two station-offline disruptions on a task with one craft job leave no station. The reference replay
is stored in each scenario's metadata. Agents follow different trajectories, so in real episodes
a disruption can still fall back or fail to fire; those cases are logged, never hidden.

## Episode (`episode.py`)

`Episode(scenario)` is the agent/harness boundary. The agent gets `call(tool, args)` and
`task_description()`; the harness gets status, `end(reason)`, tool-call records (M2), disruption
events (M3), `final_state()`, `summary()`, and `trace()` / `save_trace()`.

- **Action cap** = 2 × `reference_actions` + 5, where `reference_actions` is the optimal cost
  *with* the scheduled disruptions, i.e. what the oracle needs along the reference trajectory; it
  equals the optimum when there are no disruptions. FP2 wrote 2 × optimal + 5. Measured against
  the undisrupted optimum, that cap made the perfect oracle fail on 3 short scenarios with 3
  disruptions, where disruptions add up to 15–26 actions to a ~10-action task. This is a
  clarification to report in FP3 (D8). The cap counts every write *attempt* (successful, failed, or
  malformed `move`/`pick`/`drop`/`use`), so failing writes are not free. Read tools never count.
  The call that uses the last allowed action still runs; success on that call counts.
- **Ending**: `success`, `action_cap`, or an external `end(reason)` (e.g. `token_cap`). Afterwards
  every call returns `ERROR episode_over` and changes nothing; unreached disruptions are logged as
  `not_reached`.
- **Timing**: each record stores tool time and injector time separately, so disruption target
  selection does not inflate measured tool latency.
- Traces are deterministic apart from timing fields.

## One runtime step

1. The agent calls `call("move", {"location": "bakery"})`.
2. `tools.dispatch` validates arguments; `World.move` applies the rule → new state or an error code.
3. If a write succeeded, `ProgressTracker` checks for a completed subgoal.
4. If the subgoal count hits a scheduled trigger, `Injector` asks the solver which target breaks
   every optimal plan, mutates the world (and `map_view`, except for stale entries), and logs a
   `DisruptionEvent`.
5. The episode logs the tool call (M2) and any disruption (M3), then returns the step-2
   `ToolResult`.

## Information boundary

This boundary underpins the FP2 claim that the replanning checker sees only tool results.

| | Agent can see | Harness can see |
|---|---|---|
| World | `ToolResult.text` from the 7 tools; `read_map` reads the possibly stale `map_view` | True state, `map_view`, `final_state()` |
| Progress | `status()` job list (done/pending) | Subgoal counts, trigger points |
| Disruptions | Only their effects | Schedule, targets, before/after optimal costs |
| Ground truth | Nothing | Solver plans and optimal costs |

In Phase 2 the MCP server wraps only `Episode.call()`; everything in the right-hand column stays in
the harness process.

## Datasets

| Set | Maps | Size | Purpose |
|---|---|---|---|
| Development | 2 maps used only here | 45 (5 per cell) | Prompt development |
| Pilot | 5 test maps | 45 (5 per cell) | Floor/ceiling check, token-cap calibration |
| Test | 5 test maps | 270 (30 per cell) | Main experiment |

A cell is one (length × disruption count) combination: 3 × 3 = 9 cells.

Generation (`generator.py`, `scripts/generate_datasets.py`): within a cell, scenario *i* uses map
*i* mod (number of maps), a fixed job composition (D7), and a seed derived from
(set, length, disruptions, *i*); placement is resampled until the proven optimum falls within
±10% of the level (9–11, 18–22, 27–33). Generation rules: deliver targets are distinct and never
a craft input in the same scenario (closing the re-delivery loophole of latched completion);
craft outputs are distinct; every needed item exists exactly once; a delivery item never starts at
its own destination; 2–3 distractor items use unneeded types. Disruption types are balanced within
each cell, and triggers are evenly spaced over the scenario's subgoals. Each set has a
`manifest.json` with per-cell length, composition, rejection, and disruption-type summaries;
sets are checked to be disjoint. All 360 scenarios generate in about 7 s on 6 cores, and none was
rejected for an unproven optimum. Scenario IDs follow
`{set}_L{10|20|30}_D{0|1|3}_{index:03d}`.

## Validation (Phase 1 exit criteria)

`scripts/validate_datasets.py --set all` checks every scenario and writes
`data/validation/<set>_report.json`; it exits non-zero on any failure.

1. **Schema and map**: the scenario file and its map pass all integrity and map invariants.
2. **Solver**: solvable, optimum proven, equal to the recorded `optimal_actions`, and within ±10% of
   the length level.
3. **Schedule replay**: along the reference trajectory every disruption fires, raises the optimal
   cost, and affects at least one job, with the same targets recorded at generation.
4. **Oracle** (`oracle.py`, a full-state agent that re-plans after any failure or disruption,
   playing through `Episode.call`): succeeds within the cap using exactly `reference_actions`
   writes, with every disruption fired strictly.
5. **Unit tests** (`pytest`) cover every tool, error code, transition, and disruption type. The
   solver is checked against brute force.
6. **Human playthroughs**: about 10 scenarios played through the text interface
   (`scripts/play.py`, step 12).

Current result: **360/360 scenarios pass** (dev 45, pilot 45, test 270), in about 35 s on 6 cores.
The oracle has at least 14 write actions of headroom under the cap in every scenario.

## Invariants

1. **Determinism** — same scenario + same agent actions ⇒ identical observations and disruption
   targets. All randomness uses an explicitly seeded `random.Random`; every iteration over a set or
   dict is `sorted()` (string hash order varies between Python processes); and solver budgets count
   search expansions only, never wall-clock time, so results cannot depend on machine load
   (injector: 300k expansions, generator: 2M).
2. **Solvable by construction** — every station type appears at least twice, bridges are never
   closed, and the solver re-checks solvability after every disruption.
3. **Fair across strategies** — progress-based triggers and a shared target rule give every
   strategy equivalent disruptions.
4. **Cost = write actions** — solver optimum, action cap, and excess-action metrics all count only
   `move`, `pick`, `drop`, `use`.
5. **No silent failures** — bad input returns an error code and never raises; a disruption with no
   valid target is logged as not fired.

## Design decisions

| ID | Decision | Current choice | Status |
|---|---|---|---|
| D1 | Repo location | `finalproject/errandworld/` (renamed from `when-to-replan/`) | Adopted |
| D2 | Tooling | Python 3.11+, `pyproject.toml`, `pytest`, stdlib-only core | Adopted |
| D3 | Inventory capacity | 3 items (configurable), for realism. Benchmarks showed capacity does **not** affect solver tractability (2/3/4/6 all behave the same). Not in FP2 — report under "Changes since FP2". | Proposed (awaiting sign-off) |
| D4 | Job completion | Latched on delivery | Proposed (awaiting sign-off) |
| D5 | Map view | `read_map` serves `map_view`, synced on every disruption except stale map entry | Proposed (awaiting sign-off) |
| D6 | Task size vs. length | Length levels are **10/20/30** optimal write actions (FP2: 10/20/40), built from about **2/3/4 jobs** plus travel on sparse ring-with-chords maps (avg degree 2.4), not 3/6/12 jobs, so the solver can prove every optimum. The top level was lowered from 40 "for now" (it could return if a faster solver or a bounds-based approach is adopted). Deviation from FP2 — report under "Changes since FP2". | Adopted |
| D7 | Job composition | Fixed per scenario, not sampled: half the jobs are craft jobs (the 3-job level alternates 1 and 2 crafts). Sampling it let the ±10% length filter select it (27% craft at length 10 vs 60% at length 30), confounding length with structure. **Dependent (tier-2) jobs are off** by default, because a 2-job task cannot hold one without becoming all-craft; the code supports them (`GeneratorConfig.dependent_jobs`). FP2 said "some jobs depend on the outputs of others" — report as a deviation. | Proposed (awaiting sign-off) |
| D8 | Action cap reference | 2 × `reference_actions` + 5, where `reference_actions` = optimal cost with the scheduled disruptions (stored per scenario). FP2's 2 × optimal + 5 against the undisrupted optimum failed the oracle on short, heavily disrupted tasks. Also the baseline for FP2's excess-actions metric. **Phase 2 consequence:** the action budget now grows with disruptions (the oracle needs up to +26 actions in L10_D3), so a token cap set per *length* (FP2: 1.5× median ReAct usage) would bind harder in D3 cells than D0 cells and confound RQ1. The pilot should calibrate the token cap per *cell*, or scale it with `reference_actions`. | Proposed (awaiting sign-off) |

## Solver limits (resolved by D6)

Measured with `scripts/bench_solver.py` on ring-with-chords maps (15 locations, avg degree 2.4),
budget 10 s per solve:

| Jobs | Optimal length | Proven within 10 s |
|---|---|---|
| 2 (items close together) | median 12 | 100% (max 0.04 s) |
| 3 | median 23 | 100% (max 0.4 s) |
| 4 (50% craft) | median 29 | 95% (p95 4 s); the generator uses a 60 s offline budget |
| 5 | median 30–33 | 85–90% |
| 12 (FP2 original, denser maps) | 31–37 | 0–30% |

Inventory capacity (2/3/4/6) has no effect. Mid-episode re-solves (what the injector runs when a
disruption fires) are much faster than from-scratch solves because fewer jobs are pending.
