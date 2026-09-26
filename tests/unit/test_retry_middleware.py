import asyncio

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.middleware.retry_middleware import (
    MalformedResponseError,
    ModelRetryMiddleware,
    SanitizingModelFallbackMiddleware,
    sanitize_tool_calls,
)


def test_invalid_tool_name_retries_then_raises_malformed_response():
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        return ModelResponse(
            result=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "check_link:url_check_results",
                            "args": {},
                            "id": "call",
                        }
                    ],
                )
            ]
        )

    request = ModelRequest(model=object(), messages=[], tools=[])
    with pytest.raises(MalformedResponseError, match="invalid tool call name"):
        asyncio.run(
            ModelRetryMiddleware(max_retries=1, initial_delay=0).awrap_model_call(
                request, handler
            )
        )

    assert calls == 2


def test_sanitize_tool_calls_splits_concatenated_call():
    messages = [
        AIMessage.model_construct(
            content="",
            tool_calls=[
                {
                    "name": "check_linkurl_check_results",
                    "args": '{"url":"https://example.com"}{"results":["ok"]}',
                    "id": "malformed",
                }
            ],
            invalid_tool_calls=[],
            additional_kwargs={},
            response_metadata={},
            type="ai",
        )
    ]

    sanitized = sanitize_tool_calls(messages, {"check_link", "url_check_results"})

    calls = sanitized[0].tool_calls
    assert [call["name"] for call in calls] == ["check_link", "url_check_results"]
    assert [call["args"] for call in calls] == [
        {"url": "https://example.com"},
        {"results": ["ok"]},
    ]
    assert len({call["id"] for call in calls}) == 2
    assert messages[0].tool_calls[0]["name"] == "check_linkurl_check_results"


def test_fallback_switch_removes_poisoned_history():
    middleware = object.__new__(SanitizingModelFallbackMiddleware)
    middleware.models = [object()]
    seen_messages = []
    attempts = 0
    messages = [
        HumanMessage("check this"),
        AIMessage(
            content="",
            tool_calls=[{"name": "check_link:url_check_results", "args": {}, "id": "bad"}],
        ),
        ToolMessage(content="bad result", tool_call_id="bad"),
    ]
    request = ModelRequest(
        model=object(),
        messages=messages,
        tools=[{"name": "check_link"}, {"name": "url_check_results"}],
    )

    async def handler(fallback_request):
        nonlocal attempts
        attempts += 1
        seen_messages.append(fallback_request.messages)
        if attempts == 1:
            raise RuntimeError("primary failed")
        return ModelResponse(result=[AIMessage(content="ok")])

    result = asyncio.run(middleware.awrap_model_call(request, handler))

    assert result.result[0].content == "ok"
    assert attempts == 2
    assert all(
        not isinstance(message, ToolMessage)
        for message in seen_messages[1]
    )
    assert seen_messages[1][1].tool_calls == []
