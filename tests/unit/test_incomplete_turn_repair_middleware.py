"""Tests for repairing incomplete tool-only turns."""

from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages

from src.middleware.incomplete_turn_repair_middleware import (
    IncompleteTurnRepairMiddleware,
)


def _tool_trajectory(call_id: str = "call-1") -> list:
    return [
        AIMessage(
            content="",
            tool_calls=[{"name": "check_links", "args": {}, "id": call_id}],
        ),
        ToolMessage(content="checked", tool_call_id=call_id),
    ]


def test_repairs_orphaned_history_but_preserves_newest_turn():
    middleware = IncompleteTurnRepairMiddleware()
    old_human = HumanMessage(content="old question")
    newest_human = HumanMessage(content="new question")
    active_trajectory = _tool_trajectory("active-call")
    state = {
        "messages": [old_human, *_tool_trajectory(), newest_human, *active_trajectory]
    }

    update = middleware.before_agent(state, runtime=SimpleNamespace())
    repaired = add_messages(state["messages"], update["messages"])

    assert repaired[0] == old_human
    assert repaired[1].content == "This turn did not complete."
    assert repaired[2] == newest_human
    assert repaired[3:] == active_trajectory


def test_leaves_already_clean_history_unchanged():
    middleware = IncompleteTurnRepairMiddleware()
    state = {
        "messages": [
            HumanMessage(content="question"),
            AIMessage(content="answer"),
            HumanMessage(content="new question"),
        ]
    }

    assert middleware.before_agent(state, runtime=SimpleNamespace()) is None


def test_repair_is_idempotent():
    middleware = IncompleteTurnRepairMiddleware()
    state = {
        "messages": [
            HumanMessage(content="old question"),
            *_tool_trajectory(),
            HumanMessage(content="new question"),
        ]
    }

    update = middleware.before_agent(state, runtime=SimpleNamespace())
    repaired_state = {"messages": update["messages"][1:]}

    assert middleware.before_agent(repaired_state, runtime=SimpleNamespace()) is None
