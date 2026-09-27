from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage

from src.middleware.checkpoint_repair_middleware import (
    NO_AI_RESPONSE_MESSAGE,
    CheckpointRepairMiddleware,
)


def test_before_agent_removes_unanswered_turn_and_orphans():
    middleware = CheckpointRepairMiddleware()
    state = {
        "messages": [
            HumanMessage(content="old question", id="human-old"),
            ToolMessage(content="orphan", tool_call_id="tool-1", id="tool-old"),
            AIMessage(content="", id="ai-old"),
            HumanMessage(content="current question", id="human-current"),
        ]
    }
    runtime = SimpleNamespace(stream_writer=lambda value: runtime.events.append(value))
    runtime.events = []

    update = middleware.before_agent(state, runtime)

    assert [message.id for message in update["messages"]] == [
        "human-old",
        "tool-old",
        "ai-old",
    ]
    assert all(isinstance(message, RemoveMessage) for message in update["messages"])
    assert runtime.events == [{"checkpoint_repair_removed_messages": 3}]


def test_before_agent_preserves_completed_turns():
    middleware = CheckpointRepairMiddleware()
    state = {
        "messages": [
            HumanMessage(content="old question", id="human-old"),
            AIMessage(content="answered", id="ai-old"),
            HumanMessage(content="current question", id="human-current"),
        ]
    }

    assert middleware.before_agent(state, SimpleNamespace()) is None


def test_after_agent_adds_user_facing_fallback_without_ai_text():
    middleware = CheckpointRepairMiddleware()
    state = {"messages": [HumanMessage(content="question", id="human")]}

    update = middleware.after_agent(state, SimpleNamespace())

    assert update["messages"][0].content == NO_AI_RESPONSE_MESSAGE
