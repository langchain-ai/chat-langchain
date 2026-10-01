from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages

from src.middleware.abandoned_turn_middleware import (
    INTERRUPTED_TURN_MARKER,
    AbandonedTurnMiddleware,
)


def _tool_turn(prefix: str) -> list:
    tool_call_id = f"{prefix}-call"
    return [
        AIMessage(
            content="",
            id=f"{prefix}-ai",
            tool_calls=[
                {
                    "name": "check_links",
                    "args": {"urls": ["https://example.com"]},
                    "id": tool_call_id,
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content="partial result", id=f"{prefix}-tool", tool_call_id=tool_call_id
        ),
    ]


def _apply(state: dict, update: dict | None) -> list:
    return add_messages(state["messages"], update["messages"] if update else [])


def test_cleans_abandoned_turn_before_latest_question():
    state = {
        "messages": [
            HumanMessage(content="first question", id="h1"),
            *_tool_turn("first"),
            HumanMessage(content="new question", id="h2"),
        ]
    }

    update = AbandonedTurnMiddleware().before_agent(state, SimpleNamespace())
    messages = _apply(state, update)

    assert [message.type for message in messages] == ["human", "ai", "human"]
    assert messages[1].content == INTERRUPTED_TURN_MARKER
    assert messages[-1].id == "h2"


def test_cleans_two_consecutive_abandoned_turns():
    state = {
        "messages": [
            HumanMessage(content="first question", id="h1"),
            *_tool_turn("first"),
            HumanMessage(content="second question", id="h2"),
            *_tool_turn("second"),
            HumanMessage(content="new question", id="h3"),
        ]
    }

    update = AbandonedTurnMiddleware().before_agent(state, SimpleNamespace())
    messages = _apply(state, update)

    assert [message.type for message in messages] == [
        "human",
        "ai",
        "human",
        "ai",
        "human",
    ]
    assert messages[1].content == INTERRUPTED_TURN_MARKER
    assert messages[3].content == INTERRUPTED_TURN_MARKER
    assert messages[-1].id == "h3"


def test_leaves_clean_thread_unchanged():
    state = {
        "messages": [
            HumanMessage(content="first question", id="h1"),
            AIMessage(content="answer", id="a1"),
            HumanMessage(content="new question", id="h2"),
        ]
    }

    assert AbandonedTurnMiddleware().before_agent(state, SimpleNamespace()) is None


def test_cleanup_is_idempotent():
    state = {
        "messages": [
            HumanMessage(content="first question", id="h1"),
            *_tool_turn("first"),
            HumanMessage(content="new question", id="h2"),
        ]
    }
    middleware = AbandonedTurnMiddleware()
    cleaned = _apply(state, middleware.before_agent(state, SimpleNamespace()))

    assert middleware.before_agent({"messages": cleaned}, SimpleNamespace()) is None
