from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from src.middleware.turn_cleanup_middleware import (
    FAILED_TURN_MESSAGE,
    TurnCleanupMiddleware,
)


def test_before_agent_persists_cleanup_and_preserves_current_human_message():
    middleware = TurnCleanupMiddleware()
    state = {
        "messages": [
            AIMessage(content="Completed answer", id="completed"),
            HumanMessage(content="Earlier question", id="failed-human"),
            AIMessage(content="", id="empty-ai"),
            ToolMessage(
                content="Link Check Results: 1/1 valid",
                name="check_links",
                tool_call_id="check-links-1",
                id="check-links-1",
            ),
            HumanMessage(content="Current question", id="current-human"),
        ]
    }

    update = middleware.before_agent(state, runtime=SimpleNamespace())

    assert update is not None
    assert update["messages"][0].id == REMOVE_ALL_MESSAGES
    persisted_messages = update["messages"][1:]
    assert [message.content for message in persisted_messages] == [
        "Completed answer",
        "Earlier question",
        FAILED_TURN_MESSAGE,
        "Current question",
    ]
    assert not any(
        isinstance(message, ToolMessage) and message.name == "check_links"
        for message in persisted_messages
    )
    assert persisted_messages[-1].id == "current-human"
