from app.budget import Budget
from app.types import LLMResult, ToolResult


def llm_call(input_tokens=0, cached=0, output=0):
    return LLMResult("", input_tokens=input_tokens, cached_input_tokens=cached, output_tokens=output)


def tool_call(name, ok=True):
    return ToolResult(tool=name, args={}, ok=ok, error=None if ok else "road_closed")


def test_no_limits():
    budget = Budget()
    budget.charge_llm(llm_call(input_tokens=10_000, output=10_000))
    budget.charge_tool(tool_call("move"))
    assert budget.exceeded() is None


def test_token_count():
    budget = Budget()
    budget.charge_llm(llm_call(input_tokens=10, cached=5, output=3))
    assert budget.tokens == 18


def test_token_cap():
    budget = Budget(max_tokens=20)
    budget.charge_llm(llm_call(input_tokens=10, output=5))
    assert budget.exceeded() is None
    budget.charge_llm(llm_call(input_tokens=10, output=5))
    assert budget.exceeded() == "token_cap"


def test_only_writes_count():
    budget = Budget(max_write_actions=1)
    budget.charge_tool(tool_call("look"))
    budget.charge_tool(tool_call("status"))
    assert budget.write_actions == 0 and budget.exceeded() is None
    budget.charge_tool(tool_call("pick"))
    assert budget.exceeded() == "action_cap"


def test_failed_write_counts():
    budget = Budget(max_write_actions=1)
    budget.charge_tool(tool_call("move", ok=False))
    assert budget.exceeded() == "action_cap"


def test_step_cap():
    budget = Budget(max_steps=2)
    budget.charge_llm(llm_call())
    assert budget.exceeded() is None
    budget.charge_llm(llm_call())
    assert budget.exceeded() == "step_cap"
