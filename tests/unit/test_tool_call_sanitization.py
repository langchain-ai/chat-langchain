"""Tests for provider-safe tool-call name sanitization."""

import re
from types import SimpleNamespace

from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, ToolMessage

from src.middleware.tool_call_sanitization import (
    ToolCallSanitizationMiddleware,
    sanitize_messages,
    sanitize_tool_name,
)


def _request(messages):
    return ModelRequest(model=SimpleNamespace(), messages=messages)


def test_sanitizes_invalid_name_and_mirrored_function_call():
    ai_message = AIMessage(
        content="",
        id="ai-1",
        tool_calls=[
            {
                "name": "check_link:url_check_results",
                "args": {},
                "id": "call-1",
                "type": "tool_call",
            }
        ],
        additional_kwargs={
            "function_call": {
                "name": "check_link:url_check_results",
                "arguments": "{}",
            }
        },
    )
    tool_message = ToolMessage(
        content="check_link:url_check_results is not a valid tool",
        tool_call_id="call-1",
    )

    sanitized = sanitize_messages([ai_message, tool_message])

    assert sanitized[0].tool_calls[0]["name"] == "check_link_url_check_results"
    assert (
        sanitized[0].additional_kwargs["function_call"]["name"]
        == "check_link_url_check_results"
    )
    assert sanitized[0].tool_calls[0]["id"] == sanitized[1].tool_call_id
    assert sanitized[1] is tool_message


def test_empty_or_invalid_names_are_nonempty_and_provider_safe():
    for name in ["", ":", "é"]:
        sanitized = sanitize_tool_name(name)
        assert sanitized
        assert all(char.isascii() and (char.isalnum() or char in "_-") for char in sanitized)


def test_sanitizes_openai_bound_payload():
    ai_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "check_link:url_check_results",
                "args": {},
                "id": "call-1",
                "type": "tool_call",
            }
        ],
    )
    middleware = ToolCallSanitizationMiddleware()
    seen = {}

    def handler(request):
        seen["messages"] = request.messages
        return ModelResponse(result=[])

    middleware.wrap_model_call(_request([ai_message]), handler)

    names = [
        tool_call["name"]
        for message in seen["messages"]
        if isinstance(message, AIMessage)
        for tool_call in message.tool_calls
    ]
    assert names == ["check_link_url_check_results"]
    assert all(re.fullmatch(r"[a-zA-Z0-9_-]+", name) for name in names)


def test_sanitizes_model_output_before_checkpointing():
    ai_message = AIMessage(
        content="",
        id="ai-1",
        tool_calls=[
            {
                "name": "check_link:url_check_results",
                "args": {},
                "id": "call-1",
                "type": "tool_call",
            }
        ],
    )
    tool_message = ToolMessage(content="invalid tool", tool_call_id="call-1")
    middleware = ToolCallSanitizationMiddleware()

    update = middleware.after_model(
        {"messages": [ai_message, tool_message]}, SimpleNamespace()
    )

    assert update is not None
    assert update["messages"][0].tool_calls[0]["name"] == "check_link_url_check_results"
    assert update["messages"][0].tool_calls[0]["id"] == tool_message.tool_call_id
