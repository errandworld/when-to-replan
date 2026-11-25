"""Shared data shapes for the agent side: dataclasses and one protocol."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:  # the checker imports this module, so only type checkers import it back
    from app.checker import Trigger

# Tools that change world state. These count against the action cap; the read tools do not.
WRITE_TOOLS = ("move", "pick", "drop", "use")
READ_TOOLS = ("look", "status", "read_map")
TOOLS = WRITE_TOOLS + READ_TOOLS

# The codes a failed tool call can return.
ERROR_CODES = (
    "not_adjacent",
    "road_closed",
    "item_not_here",
    "not_carrying",
    "not_at_station",
    "station_offline",
    "missing_items",
    "invalid_recipe",
)

StopReason = Literal[
    "goal_reached",         
    "agent_stopped",         # the agent replied without calling a tool
    "token_cap",
    "action_cap",
    "step_cap",
    "infrastructure_error",  # API or server failure
    "loop_error",            # unknown tool, unparseable reply
]


@dataclass
class ToolCall:
    """A call the model asked for. The id is the provider's, used to reply to it."""

    name: str
    args: dict[str, Any] = field(default_factory=dict)
    id: str = ""


@dataclass
class ToolResult:
    """The return of one tool call. The payload is whatever the server sent back."""

    tool: str
    args: dict[str, Any]
    ok: bool
    error: str | None = None  # one of ERROR_CODES when ok is False
    payload: dict[str, Any] = field(default_factory=dict)
    latency_s: float = 0.0
    text: str = ""  # what the model reads; empty for stand-ins that only have a payload

    @property
    def is_write(self) -> bool:
        return self.tool in WRITE_TOOLS


@dataclass
class LLMResult:
    """One model call. Cached and uncached input are counted separately because they are
    priced separately. The model version is the one the provider reported."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0  # uncached
    cached_input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    model_version: str = ""


# The plan-based loops: "remaining" is adaptive replanning, "step" is local repair.
RevisionScope = Literal["remaining", "step"]
PlanKind = Literal["initial", "rewrite", "repair"]


@dataclass
class PlanVersion:
    """One version of the plan (M4). `steps` is the whole plan as it stood once this version
    was made, each step as a dict, and `start` the first step it changed. `llm_calls` indexes
    the calls in EpisodeOutcome.llm_calls that produced it, a retry included."""

    version: int
    kind: PlanKind
    start: int
    steps: list[dict[str, Any]]
    reasoning: str = ""
    warnings: list[str] = field(default_factory=list)
    llm_calls: list[int] = field(default_factory=list)


@dataclass
class RevisionEvent:
    """One replan or repair (M4): what set it off, and the plan version it produced.
    `write_cap` is the repair cap applied, and `cut` the steps the cap removed."""

    trigger: Trigger
    scope: RevisionScope
    plan_version: int
    write_cap: int | None = None
    cut: int = 0


@dataclass
class EpisodeOutcome:
    """What a loop returns when an episode stops, the two lists holding its calls in order.
    Success is decided by the orchestrator, which reads the final world state; a loop leaves
    it None."""

    stop_reason: StopReason
    final_output: str = ""
    llm_calls: list[LLMResult] = field(default_factory=list)
    tool_calls: list[ToolResult] = field(default_factory=list)
    wall_clock_s: float = 0.0
    error: str | None = None
    success: bool | None = None
    # Filled by the plan-based loops only; ReAct leaves them empty.
    plans: list[PlanVersion] = field(default_factory=list)
    revisions: list[RevisionEvent] = field(default_factory=list)


@runtime_checkable
class ToolClient(Protocol):
    """A tool end: the definitions handed to the model, and one call at a time."""

    def definitions(self) -> list[dict[str, Any]]: ...

    def call(self, name: str, args: dict[str, Any]) -> ToolResult: ...
