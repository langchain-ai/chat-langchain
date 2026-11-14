from unittest.mock import MagicMock

from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.middleware.round_budget_middleware import RoundBudgetMiddleware, count_tool_rounds


def _rounds(n: int, start: int = 0) -> list:
    out = []
    for i in range(start, start + n):
        out.append(AIMessage(content="", tool_calls=[{"name": "search", "args": {}, "id": f"c{i}"}]))
        out.append(ToolMessage(content="r", tool_call_id=f"c{i}"))
    return out


def _request(messages, tools=("t",)):
    return ModelRequest(
        model=MagicMock(), messages=messages, system_message=SystemMessage(content="base"),
        tool_choice=None, tools=list(tools), response_format=None, state={}, runtime=None,
    )


def _run(mw, req):
    seen = {}
    def handler(r):
        seen["r"] = r
        return "ok"
    mw.wrap_model_call(req, handler)
    return seen["r"]


def test_counts_only_current_turn():
    msgs = [HumanMessage("a"), *_rounds(6), AIMessage("answer"), HumanMessage("b"), *_rounds(2, 10)]
    assert count_tool_rounds(msgs) == 2


def test_under_budget_unchanged():
    req = _request([HumanMessage("q"), *_rounds(3)])
    out = _run(RoundBudgetMiddleware(), req)
    assert out.system_message.content == "base" and out.tools == ["t"]


def test_nudge_after_four_rounds_keeps_tools():
    out = _run(RoundBudgetMiddleware(), _request([HumanMessage("q"), *_rounds(4)]))
    assert "Research budget" in out.system_message.content and out.tools == ["t"]
    assert out.system_message.content.startswith("base")


def test_stop_after_eight_rounds_removes_tools():
    out = _run(RoundBudgetMiddleware(), _request([HumanMessage("q"), *_rounds(8)]))
    assert "no longer available" in out.system_message.content and out.tools == []


def test_list_content_system_message():
    req = _request([HumanMessage("q"), *_rounds(5)])
    req = req.override(system_message=SystemMessage(content=[{"type": "text", "text": "base"}]))
    out = _run(RoundBudgetMiddleware(), req)
    assert out.system_message.content[0]["text"] == "base" and len(out.system_message.content) == 2
