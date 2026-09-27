from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.base import create_checkpoint, empty_checkpoint, uuid6
from langgraph.checkpoint.memory import InMemorySaver

from src.middleware.failure_repair_middleware import (
    _FAILURE_MESSAGE,
    _repair_checkpoint,
)


def _save_messages(checkpointer, messages):
    config = {"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}}
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = {"messages": messages}
    checkpoint["channel_versions"] = {
        "messages": checkpointer.get_next_version(None, None)
    }
    checkpoint = create_checkpoint(checkpoint, channels=None, step=0, id=str(uuid6()))
    checkpointer.put(
        config,
        checkpoint,
        {"source": "input", "step": 0},
        {"messages": checkpoint["channel_versions"]["messages"]},
    )
    return config


def test_repair_removes_aborted_tool_messages_and_persists_failure():
    checkpointer = InMemorySaver()
    config = _save_messages(
        checkpointer,
        [
            HumanMessage(content="Earlier"),
            AIMessage(content="Answered"),
            HumanMessage(content="Current question"),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "search", "args": {}, "id": "call-1", "type": "tool_call"}
                ],
            ),
            ToolMessage(content="partial", tool_call_id="call-1"),
        ],
    )

    assert _repair_checkpoint(checkpointer, config)

    messages = checkpointer.get_tuple(config).checkpoint["channel_values"]["messages"]
    assert messages[-2].content == "Current question"
    assert messages[-1].content == _FAILURE_MESSAGE
    assert not any(isinstance(message, ToolMessage) for message in messages)
    assert not any(getattr(message, "tool_calls", None) for message in messages)


def test_repair_is_idempotent():
    checkpointer = InMemorySaver()
    config = _save_messages(checkpointer, [HumanMessage(content="Current question")])

    assert _repair_checkpoint(checkpointer, config)
    assert not _repair_checkpoint(checkpointer, config)
