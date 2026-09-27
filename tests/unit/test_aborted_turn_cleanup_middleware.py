"""Tests for repairing incomplete checkpointed turns."""

from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages

from src.middleware.aborted_turn_cleanup_middleware import (
    ABORTED_TURN_MESSAGE,
    AbortedTurnCleanupMiddleware,
)


def apply_update(messages, update):
    return add_messages(messages, update["messages"])


def test_before_agent_repairs_repeated_check_links_pairs():
    middleware = AbortedTurnCleanupMiddleware()
    messages = [
        HumanMessage(content="first", id="h1"),
        AIMessage(
            content="",
            tool_calls=[{"name": "check_links", "args": {}, "id": "call-1"}],
            id="a1",
        ),
        ToolMessage(content="same", tool_call_id="call-1", id="t1"),
        AIMessage(
            content="",
            tool_calls=[{"name": "check_links", "args": {}, "id": "call-2"}],
            id="a2",
        ),
        ToolMessage(content="same", tool_call_id="call-2", id="t2"),
        HumanMessage(content="second", id="h2"),
    ]

    update = middleware.before_agent(messages_state(messages), SimpleNamespace())
    repaired = apply_update(messages, update)

    assert [message.type for message in repaired] == ["human", "ai", "human"]
    assert repaired[1].content == ABORTED_TURN_MESSAGE


def test_before_agent_preserves_well_formed_multi_turn_state():
    middleware = AbortedTurnCleanupMiddleware()
    messages = [
        HumanMessage(content="first", id="h1"),
        AIMessage(content="answer", id="a1"),
        HumanMessage(content="second", id="h2"),
        AIMessage(content="answer two", id="a2"),
    ]

    assert middleware.before_agent(messages_state(messages), SimpleNamespace()) is None


def test_before_agent_does_not_repair_current_final_human_message():
    middleware = AbortedTurnCleanupMiddleware()
    messages = [HumanMessage(content="current", id="h1")]

    assert middleware.before_agent(messages_state(messages), SimpleNamespace()) is None


def test_model_failure_returns_terminating_assistant_message():
    middleware = AbortedTurnCleanupMiddleware()

    def failing_handler(request):
        raise TimeoutError("provider timed out")

    response = middleware.wrap_model_call(SimpleNamespace(), failing_handler)

    assert isinstance(response, AIMessage)
    assert response.content == ABORTED_TURN_MESSAGE


def messages_state(messages):
    return {"messages": messages}
