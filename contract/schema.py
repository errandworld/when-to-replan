"""contract/schema.py: What the agent side and measurement side can rely on."""
### JUST A PROTOTYPE! NOT A FINAL VERSION! The schema is likely to change, and the server will not be backward compatible.

from typing import Literal, Optional
from pydantic import BaseModel

ErrorCode = Literal["not_adjacent", "road_closed", "item_not_here", "not_carrying",
                    "not_at_station", "station_offline", "missing_items", "invalid_recipe"]

class ToolResult(BaseModel):
    ok: bool
    error: Optional[ErrorCode] = None
    payload: dict = {}          # one of the payload models below

class ItemView(BaseModel):    id: str; type: str
class StationView(BaseModel): id: str; type: str; online: bool
class RoadView(BaseModel):    to: str; open: bool

class MovePayload(BaseModel):   location: str; adjacent: list[str]
class LookPayload(BaseModel):   location: str; items: list[ItemView]; stations: list[StationView]; roads: list[RoadView]
class PickPayload(BaseModel):   inventory: list[str]
class DropPayload(BaseModel):   inventory: list[str]; completed_job: Optional[str] = None
class UsePayload(BaseModel):    output_item: str
class StatusPayload(BaseModel): location: str; inventory: list[str]; jobs: dict  # job id -> "done" | "pending"
class ReadMapPayload(BaseModel):  # same shape fresh or stale
    roads: list[dict]; items: list[dict]; stations: list[dict]

class CallRecord(BaseModel):       # one line per tool call in server_events.jsonl (M2)
    call_index: int; tool: str; args: dict; ok: bool
    error: Optional[ErrorCode] = None
    resource_touched: Optional[str] = None
    valid: bool                    # equals ok for now; kept for M2

class DisruptionRecord(BaseModel): # one line per fired disruption (M3)
    disruption_id: str; type: str; scheduled_after_subgoals: int
    fired_after_call_index: int; target: str
    announced: bool; affected_jobs: list[str]