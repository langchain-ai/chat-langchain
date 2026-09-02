import asyncio
from types import SimpleNamespace

from langchain_core.messages import ToolMessage

from src.middleware.tool_retry_middleware import ToolRetryMiddleware


def _request():
    return SimpleNamespace(
        tool_call={"name": "search_docs", "id": "call-1"},
    )


def test_transient_tool_message_retries_and_returns_error_envelope():
    attempts = 0

    async def handler(_request):
        nonlocal attempts
        attempts += 1
        return ToolMessage(
            content=[{"type": "text", "text": "Search failed: 504 Gateway Time-out"}],
            name="search_docs",
            tool_call_id="call-1",
        )

    middleware = ToolRetryMiddleware(
        max_attempts=3,
        initial_delay=0,
        backoff_factor=1,
    )
    result = asyncio.run(middleware.awrap_tool_call(_request(), handler))

    assert attempts == 3
    assert isinstance(result, ToolMessage)
    assert '"error": "Tool unavailable"' in result.content
    assert "Search failed: 504 Gateway Time-out" in result.content


def test_successful_tool_message_is_returned_without_retry():
    attempts = 0
    expected = ToolMessage(
        content="Documentation result",
        name="search_docs",
        tool_call_id="call-1",
    )

    async def handler(_request):
        nonlocal attempts
        attempts += 1
        return expected

    middleware = ToolRetryMiddleware(initial_delay=0)
    result = asyncio.run(middleware.awrap_tool_call(_request(), handler))

    assert attempts == 1
    assert result is expected
