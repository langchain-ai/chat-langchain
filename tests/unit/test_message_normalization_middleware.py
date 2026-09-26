from langchain_core.messages import AIMessage

from src.middleware.message_normalization_middleware import (
    MessageNormalizationMiddleware,
)


def test_removes_legacy_function_call_for_parallel_tool_calls():
    message = AIMessage(
        content="",
        additional_kwargs={"function_call": {"name": "corrupt"}, "other": "kept"},
        tool_calls=[
            {"name": "first_tool", "args": {}, "id": "call-1", "type": "tool_call"},
            {"name": "second_tool", "args": {}, "id": "call-2", "type": "tool_call"},
        ],
    )

    update = MessageNormalizationMiddleware().before_model(
        {"messages": [message]}, runtime=None
    )

    normalized = update["messages"][0]
    assert "function_call" not in normalized.additional_kwargs
    assert normalized.additional_kwargs["other"] == "kept"
    assert normalized.tool_calls == message.tool_calls


def test_preserves_legacy_function_call_for_single_tool_call():
    message = AIMessage(
        content="",
        additional_kwargs={"function_call": {"name": "valid"}},
        tool_calls=[
            {"name": "only_tool", "args": {}, "id": "call-1", "type": "tool_call"}
        ],
    )

    middleware = MessageNormalizationMiddleware()

    assert middleware.before_model({"messages": [message]}, runtime=None) is None
