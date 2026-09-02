import asyncio
import json

from langchain_core.messages import ToolMessage

from src.middleware.tool_retry_middleware import ToolRetryMiddleware


def _request():
    return type("Request", (), {"tool_call": {"name": "docs_search", "id": "call-1"}})()


def test_retries_tool_message_failure_and_returns_unavailable_payload():
    middleware = ToolRetryMiddleware(max_attempts=3, initial_delay=0)
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        return ToolMessage(
            content=[{"type": "text", "text": "Search failed: 504 Gateway Time-out"}],
            tool_call_id="call-1",
        )

    result = asyncio.run(middleware.awrap_tool_call(_request(), handler))

    assert calls == 3
    assert json.loads(result.content)["error"] == "Tool unavailable"


def test_does_not_retry_normal_docs_page_containing_timeout():
    middleware = ToolRetryMiddleware(max_attempts=3, initial_delay=0)
    calls = 0
    body = "The timeout parameter controls how long the client waits."

    async def handler(request):
        nonlocal calls
        calls += 1
        return ToolMessage(content=body, tool_call_id="call-1")

    result = asyncio.run(middleware.awrap_tool_call(_request(), handler))

    assert calls == 1
    assert result.content == body


def test_returns_success_after_tool_message_failure():
    middleware = ToolRetryMiddleware(max_attempts=3, initial_delay=0)
    responses = [
        ToolMessage(
            content="Docs filesystem query failed: 504 Gateway Time-out",
            tool_call_id="call-1",
        ),
        ToolMessage(content="Successful docs body", tool_call_id="call-1"),
    ]

    async def handler(request):
        return responses.pop(0)

    result = asyncio.run(middleware.awrap_tool_call(_request(), handler))

    assert result.content == "Successful docs body"


def test_returns_successful_non_tool_message_unchanged():
    middleware = ToolRetryMiddleware(max_attempts=3, initial_delay=0)
    result_value = object()

    async def handler(request):
        return result_value

    result = asyncio.run(middleware.awrap_tool_call(_request(), handler))

    assert result is result_value
