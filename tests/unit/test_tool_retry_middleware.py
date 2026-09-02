import asyncio
import json
from types import SimpleNamespace

from langchain_core.messages import ToolMessage

from src.middleware.tool_retry_middleware import ToolRetryMiddleware


def _request():
    return SimpleNamespace(tool_call={"id": "call-1", "name": "search_docs"})


def test_returned_block_error_becomes_tool_unavailable():
    middleware = ToolRetryMiddleware(initial_delay=0, returned_error_attempts=2)
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        return ToolMessage(
            content=[{"type": "text", "text": "Search failed: 504 Gateway Time-out"}],
            name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"],
            status="error",
        )

    result = asyncio.run(middleware.awrap_tool_call(_request(), handler))

    assert calls == 2
    payload = json.loads(result.content)
    assert payload["error"] == "Tool unavailable"
    assert result.content != "Search failed: 504 Gateway Time-out"


def test_returned_plain_string_error_becomes_tool_unavailable():
    middleware = ToolRetryMiddleware(initial_delay=0, returned_error_attempts=1)

    async def handler(request):
        return ToolMessage(
            content="Search failed: 504 Gateway Time-out",
            name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"],
            status="error",
        )

    result = asyncio.run(middleware.awrap_tool_call(_request(), handler))

    payload = json.loads(result.content)
    assert payload["error"] == "Tool unavailable"
    assert payload["details"] == "Search failed: 504 Gateway Time-out"


def test_successful_result_passes_through_untouched():
    middleware = ToolRetryMiddleware()
    expected = ToolMessage(
        content="Documentation result",
        name="search_docs",
        tool_call_id="call-1",
    )

    async def handler(request):
        return expected

    result = asyncio.run(middleware.awrap_tool_call(_request(), handler))

    assert result is expected


def test_returned_no_results_are_normalized():
    middleware = ToolRetryMiddleware()

    async def handler(request):
        return ToolMessage(
            content=[{"content": "No results found."}],
            name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"],
        )

    result = asyncio.run(middleware.awrap_tool_call(_request(), handler))

    assert result.content == "No results found."
