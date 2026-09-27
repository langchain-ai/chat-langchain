from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages

from src.middleware.dangling_turn_middleware import DanglingTurnMiddleware


def test_before_agent_removes_abandoned_turn_and_preserves_latest_question():
    messages = [
        HumanMessage(content="Question A", id="human-a"),
        AIMessage(content="", id="ai-a"),
        ToolMessage(content="tool result", tool_call_id="call-a", id="tool-a"),
        HumanMessage(content="Question B", id="human-b"),
    ]

    update = DanglingTurnMiddleware().before_agent(
        {"messages": messages}, runtime=SimpleNamespace()
    )

    assert update is not None
    assert [message.id for message in update["messages"]] == [
        "human-a",
        "ai-a",
        "tool-a",
    ]
    remaining = add_messages(messages, update["messages"])
    assert [message.id for message in remaining] == ["human-b"]


def test_before_agent_preserves_turn_with_prose_ai_answer():
    messages = [
        HumanMessage(content="Question A", id="human-a"),
        AIMessage(content="Here is the answer.", id="ai-a"),
        HumanMessage(content="Question B", id="human-b"),
    ]

    assert (
        DanglingTurnMiddleware().before_agent(
            {"messages": messages}, runtime=SimpleNamespace()
        )
        is None
    )
