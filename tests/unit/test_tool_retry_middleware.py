import asyncio
from types import SimpleNamespace

from langchain_core.messages import ToolMessage

from src.middleware.tool_retry_middleware import ToolRetryMiddleware


def _request():
    return SimpleNamespace(tool_call={"name": "search_docs", "id": "call-1"})


def test_mcp_failure_content_retries_and_returns_structured_error():
    calls = 0
    middleware = ToolRetryMiddleware(
        max_attempts=3, initial_delay=0, per_attempt_timeout=1
    )

    async def handler(request):
        nonlocal calls
        calls += 1
        return ToolMessage(
            content=[{"text": "Search failed: 504 Gateway Time-out", "type": "text"}],
            name="search_docs",
            tool_call_id="call-1",
        )

    result = asyncio.run(middleware.awrap_tool_call(_request(), handler))

    assert calls == 3
    assert result.content.startswith('{"error": "Tool unavailable"')


def test_successful_result_passes_through_untouched():
    middleware = ToolRetryMiddleware(per_attempt_timeout=1)
    expected = ToolMessage(
        content="documentation result", name="search_docs", tool_call_id="call-1"
    )

    async def handler(request):
        return expected

    result = asyncio.run(middleware.awrap_tool_call(_request(), handler))

    assert result is expected


def test_no_results_result_is_normalized_without_retry():
    calls = 0
    middleware = ToolRetryMiddleware(
        max_attempts=3, initial_delay=0, per_attempt_timeout=1
    )

    async def handler(request):
        nonlocal calls
        calls += 1
        return ToolMessage(
            content=[{"text": "No results found", "type": "text"}],
            name="search_docs",
            tool_call_id="call-1",
        )

    result = asyncio.run(middleware.awrap_tool_call(_request(), handler))

    assert calls == 1
    assert result.content == "No results found."


def test_handler_timeout_is_retried():
    calls = 0
    middleware = ToolRetryMiddleware(
        max_attempts=2, initial_delay=0, per_attempt_timeout=0.01
    )

    async def handler(request):
        nonlocal calls
        calls += 1
        await asyncio.sleep(1)

    result = asyncio.run(middleware.awrap_tool_call(_request(), handler))

    assert calls == 2
    assert result.content.startswith('{"error": "Tool unavailable"')
