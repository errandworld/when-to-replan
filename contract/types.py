"""contract/types.py: the A1/A2 agreement. Change only with a CHANGELOG entry."""
#JUST A PROTOTYPE! NOT A FINAL VERSION! The schema is likely to change, and the server will not be backward compatible.


from dataclasses import dataclass
from typing import Optional, Union

# IDs are strings: "loc_03", "bulb_2", "oven_1", "job_1"

@dataclass(frozen=True)
class State:
    location: str
    item_loc: frozenset          # {(item_id, "loc_02" | "inv" | "gone"), ...}
    closed_roads: frozenset      # {("loc_02", "loc_03"), ...}, sorted pairs
    offline_stations: frozenset  # {"oven_1", ...}
    jobs_done: frozenset         # {"job_1", ...}
    # TODO(decide): progress for dependent jobs. Map view stays out of State.

@dataclass(frozen=True)
class Move: to: str
@dataclass(frozen=True)
class Pick: item: str
@dataclass(frozen=True)
class Drop: item: str
@dataclass(frozen=True)
class Use:  station: str; items: tuple
@dataclass(frozen=True)
class Look:  station: str; items: tuple
Action = Union[Move, Pick, Drop, Use, Look]

@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    seed: int
    start: str
    roads: frozenset
    stations: tuple     # (id, type, location)
    items: tuple        # (id, type, location)
    jobs: tuple         # id, template, inputs, station_type, destination, requires
    disruptions: tuple  # (id, after_subgoals, type), no target
    optimal_actions: int
    task_text: str

@dataclass(frozen=True)
class Target:
    resource_id: str
    kind: str           # "road" | "station" | "item"
    affected_jobs: tuple

def initial_state(scenario: Scenario) -> State:
    """A1. Scenario in, starting State out."""
    raise NotImplementedError

def legal_successors(state: State, scenario: Scenario) -> list:
    """A1. (Action, State) pairs, write actions only."""
    raise NotImplementedError

def is_goal(state: State, scenario: Scenario) -> bool:
    """A1. True exactly when every job's goal holds."""
    raise NotImplementedError

def completed_subgoals(state: State, scenario: Scenario) -> int:
    """A1. Count of picks, uses and deliveries done."""
    raise NotImplementedError

def is_solvable(state: State, scenario: Scenario) -> bool:
    """A2. Runs inside the server at fire time."""
    raise NotImplementedError

def choose_target(state: State, scenario: Scenario, dtype: str,
                  disruption_id: str) -> Optional[Target]:
    """A2. Resource every optimal plan needs, or None. Seeded tie-break."""
    raise NotImplementedError