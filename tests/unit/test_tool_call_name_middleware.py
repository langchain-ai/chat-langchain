"""Tests for model-emitted tool-call name sanitization."""

import asyncio

from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.middleware.tool_call_name_middleware import ToolCallNameMiddleware


def _request(messages):
    return ModelRequest(
        model=object(),
        messages=messages,
        tools=[{"type": "function", "function": {"name": "check_links"}}],
    )


def test_repairs_control_tokens_and_prose_preserving_call_data():
    malformed_name = (
        "reasoning " + "x" * 53 + "<ctrl123><|assistant|> call:default_api:check_links"
    )
    assert len(malformed_name) == 114
    call = {
        "name": malformed_name,
        "args": {"urls": ["https://example.com"]},
        "id": "c1",
    }
    middleware = ToolCallNameMiddleware()
    seen = []

    async def handler(request):
        seen.append(request)
        return ModelResponse(result=[AIMessage(content="", tool_calls=[call])])

    result = asyncio.run(
        middleware.awrap_model_call(
            _request([HumanMessage(content="Check this link.")]), handler
        )
    )

    assert len(seen) == 1
    assert result.result[0].tool_calls == [
        {"name": "check_links", "args": call["args"], "id": "c1", "type": "tool_call"}
    ]


def test_drops_unrecoverable_call_and_retries_without_orphan_tool_message():
    middleware = ToolCallNameMiddleware()
    calls = []
    malformed = AIMessage(
        content="",
        tool_calls=[{"name": "not-a-bound-tool", "args": {}, "id": "bad-1"}],
    )

    async def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return ModelResponse(
                result=[
                    malformed,
                    ToolMessage(content="should be removed", tool_call_id="bad-1"),
                ]
            )
        return ModelResponse(result=[AIMessage(content="Recovered")])

    result = asyncio.run(
        middleware.awrap_model_call(
            _request([HumanMessage(content="Use a tool.")]), handler
        )
    )

    assert len(calls) == 2
    retry_messages = calls[1].messages
    assert not any(isinstance(message, ToolMessage) for message in retry_messages)
    assert isinstance(retry_messages[-1], HumanMessage)
    assert "check_links" in retry_messages[-1].content
    assert result.result[0].content == "Recovered"


def test_valid_tool_call_passes_through_unchanged():
    call = {"name": "check_links", "args": {"urls": []}, "id": "valid-1"}
    middleware = ToolCallNameMiddleware()
    seen = []

    async def handler(request):
        seen.append(request)
        return ModelResponse(result=[AIMessage(content="", tool_calls=[call])])

    result = asyncio.run(
        middleware.awrap_model_call(
            _request([HumanMessage(content="Check links.")]), handler
        )
    )

    assert len(seen) == 1
    assert result.result[0].tool_calls[0]["name"] == "check_links"
    assert result.result[0].tool_calls[0]["args"] == {"urls": []}
    assert result.result[0].tool_calls[0]["id"] == "valid-1"
