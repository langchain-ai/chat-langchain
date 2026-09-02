"""Tests for tool retry middleware failure handling."""

import asyncio
from types import SimpleNamespace

from langchain_core.messages import ToolMessage

from src.middleware.tool_retry_middleware import ToolRetryMiddleware


def test_retries_content_level_failure_and_normalizes_final_error():
    middleware = ToolRetryMiddleware(max_attempts=3, initial_delay=0)
    request = SimpleNamespace(
        tool_call={"name": "search_docs", "id": "call-1"},
    )
    calls = 0

    async def handler(_request):
        nonlocal calls
        calls += 1
        return ToolMessage(
            content=[{"type": "text", "text": "Search failed: 504 Gateway Time-out"}],
            name="search_docs",
            tool_call_id="call-1",
        )

    result = asyncio.run(middleware.awrap_tool_call(request, handler))

    assert calls == middleware.max_attempts
    assert "Tool unavailable" in result.content
