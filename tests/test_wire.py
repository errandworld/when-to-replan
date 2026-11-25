"""The server's wire format must feed the rule-based checker without translation by the loops."""
from app.checker import Beliefs
from app.plan import Precondition
from app.types import ToolResult
from errandworld.episode import Episode
from errandworld.scenario_io import load_scenario
from errandworld.wire import to_wire
 
from tests.helpers import DEV
 
 
def as_client_result(episode, tool, args=None):
    """What MCPToolClient makes of the server's reply."""
    wire = to_wire(episode.call(tool, args or {}))
    return ToolResult(tool, args or {}, ok=wire["ok"], error=wire["error"], payload=wire["data"], text=wire["text"])
 
 
def test_envelope_has_the_same_four_fields_for_every_call():
    ep = Episode(load_scenario(DEV / "dev_L10_D0_000.json"))
    for tool, args in [("look", {}), ("status", {}), ("read_map", {}), ("move", {"location": "nowhere"}), ("bogus", {})]:
        wire = to_wire(ep.call(tool, args))
        assert set(wire) == {"ok", "error", "text", "data"}
        assert isinstance(wire["text"], str) and wire["text"]
 
 
def test_failures_carry_the_error_code_and_no_data():
    ep = Episode(load_scenario(DEV / "dev_L10_D0_000.json"))
    wire = to_wire(ep.call("move", {"location": "nowhere"}))
    assert wire["ok"] is False and wire["error"] == "unknown_id" and wire["data"] == {}
 
 
def test_checker_learns_the_world_from_wire_data():
    ep = Episode(load_scenario(DEV / "dev_L10_D0_000.json"))
    beliefs = Beliefs()
    look = as_client_result(ep, "look")
    beliefs.update(look)
    item = look.payload["items"][0]
    here = look.payload["location"]
    assert beliefs.holds(Precondition("at", {"location": here})) is True
    assert beliefs.holds(Precondition("item_at", {"item": item, "location": here})) is True
    beliefs.update(as_client_result(ep, "pick", {"item": item}))
    assert beliefs.holds(Precondition("carrying", {"item": item})) is True
    assert beliefs.holds(Precondition("item_at", {"item": item, "location": here})) is False
    beliefs.update(as_client_result(ep, "read_map"))  # roads, items and stations in the checker's names
    road = look.payload["roads"][0]["to"]
    assert beliefs.holds(Precondition("road_open", {"from": here, "to": road})) is True