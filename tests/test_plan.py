import json

import pytest

from app.plan import PRECONDITION_FIELDS, Plan, PlanError, PlanStep, Precondition, parse_plan, plan_schema
from tests.fakes import FakeToolClient

TOOLS = FakeToolClient().definitions()


def reply(*steps, reasoning="Fetch the bulb, then deliver it."):
    return json.dumps({"reasoning": reasoning, "steps": list(steps)})


def step(tool, args=None, preconditions=None):
    return {"tool": tool, "args": args or {}, "preconditions": preconditions or []}


def test_parses_steps_and_preconditions():
    plan = parse_plan(
        reply(
            step("move", {"location": "shop"}, [{"type": "at", "location": "home"},
                                                {"type": "road_open", "from": "home", "to": "shop"}]),
            step("pick", {"item": "bulb"}, [{"type": "item_at", "item": "bulb", "location": "shop"}]),
        ),
        TOOLS,
    )
    assert plan.reasoning == "Fetch the bulb, then deliver it."
    assert [s.tool for s in plan.steps] == ["move", "pick"]
    assert plan.steps[0].args == {"location": "shop"}
    assert [str(p) for p in plan.steps[0].preconditions] == ["at(home)", "road_open(home, shop)"]
    assert plan.steps[1].preconditions[0] == Precondition("item_at", {"item": "bulb", "location": "shop"})
    assert plan.warnings == []


def test_round_trip():
    plan = Plan(
        steps=[PlanStep("use", {"station": "repair_1", "items": "lamp"},
                        [Precondition("station_online", {"station": "repair_1"})])],
        reasoning="r",
    )
    assert parse_plan(json.dumps(plan.to_dict()), TOOLS).to_dict() == plan.to_dict()


def test_empty_plan_is_valid():
    assert parse_plan('{"steps": []}', TOOLS).steps == []


def test_missing_preconditions_means_none():
    plan = parse_plan('{"steps": [{"tool": "look", "args": {}}]}', TOOLS)
    assert plan.steps[0].preconditions == []


@pytest.mark.parametrize(
    "text",
    [
        "```json\n" + reply(step("look")) + "\n```",
        "Here is the plan:\n" + reply(step("look")) + "\nGood luck.",
    ],
)
def test_tolerates_fences_and_prose(text):
    assert [s.tool for s in parse_plan(text, TOOLS).steps] == ["look"]


@pytest.mark.parametrize(
    "text, message",
    [
        ("not json at all", "not valid JSON"),
        ('{"plan": []}', '"steps" list'),
        ('[{"tool": "look"}]', '"steps" list'),
        (reply("look"), "step 0: expected an object"),
        (reply(step("look"), step("teleport")), "step 1: unknown tool 'teleport'"),
        (reply({"tool": "look", "args": "x"}), "args must be an object"),
        (reply(step("move")), "move needs location"),
        (reply(step("move", {"location": ""})), "move needs location"),
        (reply(step("use", {"station": "repair_1"})), "use needs items"),
        (reply({"tool": "look", "args": {}, "preconditions": "at(home)"}), "preconditions must be a list"),
        (reply(step("look", preconditions=["at(home)"])), "each precondition must be an object"),
        (reply(step("look", preconditions=[{"type": "sunny"}])), "unknown precondition type 'sunny'"),
        (reply(step("look", preconditions=[{"type": "item_at", "item": "bulb"}])), "item_at needs a location"),
    ],
)
def test_rejects_invalid_plans(text, message):
    with pytest.raises(PlanError, match=message):
        parse_plan(text, TOOLS)


def test_plan_error_is_a_value_error():
    assert issubclass(PlanError, ValueError)


def test_drops_arguments_the_tool_does_not_take():
    plan = parse_plan(reply(step("move", {"location": "shop", "item": "", "station": "repair_1"})), TOOLS)
    assert plan.steps[0].args == {"location": "shop"}
    assert plan.warnings == ["step 0: dropped arguments ['station'], which move does not take"]


@pytest.mark.parametrize(
    "tool, args, own_effect, kept",
    [
        ("move", {"location": "shop"}, {"type": "at", "location": "shop"}, {"type": "at", "location": "home"}),
        ("pick", {"item": "bulb"}, {"type": "carrying", "item": "bulb"}, {"type": "carrying", "item": "lamp"}),
    ],
)
def test_drops_preconditions_that_are_the_steps_own_effect(tool, args, own_effect, kept):
    plan = parse_plan(reply(step(tool, args, [own_effect, kept])), TOOLS)
    assert [p.to_dict() for p in plan.steps[0].preconditions] == [kept]
    assert len(plan.warnings) == 1 and "achieves rather than needs" in plan.warnings[0]


def test_strips_precondition_values():
    plan = parse_plan(reply(step("look", preconditions=[{"type": "at", "location": " shop "}])), TOOLS)
    assert plan.steps[0].preconditions[0].args == {"location": "shop"}


def test_schema_follows_the_tool_definitions():
    schema = plan_schema(TOOLS)
    step_schema = schema["properties"]["steps"]["items"]
    assert step_schema["properties"]["tool"]["enum"] == [t["function"]["name"] for t in TOOLS]
    assert set(step_schema["properties"]["args"]["properties"]) == {"location", "item", "station", "items"}
    precondition = step_schema["properties"]["preconditions"]["items"]
    assert precondition["properties"]["type"]["enum"] == list(PRECONDITION_FIELDS)
    assert {"location", "item", "from", "to", "station"} <= set(precondition["properties"])
    assert schema["required"] == ["steps"]
