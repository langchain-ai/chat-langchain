import json
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import ToolMessage

from src.middleware.tool_retry_middleware import ToolRetryMiddleware


@pytest.fixture
def tool_request():
    return type(
        "Request",
        (),
        {"tool_call": {"name": "search_docs_by_lang_chain", "id": "call-1"}},
    )()


@pytest.mark.asyncio
async def test_retries_retryable_tool_message_and_returns_unavailable(tool_request):
    middleware = ToolRetryMiddleware(max_attempts=3, initial_delay=0)
    handler = AsyncMock(
        return_value=ToolMessage(
            content=[{"type": "text", "text": "Search failed: 504 Gateway Time-out"}],
            tool_call_id="call-1",
        )
    )

    result = await middleware.awrap_tool_call(tool_request, handler)

    assert handler.await_count == 3
    assert json.loads(result.content)["error"] == "Tool unavailable"
    assert "504 Gateway Time-out" in json.loads(result.content)["details"]


@pytest.mark.asyncio
async def test_returns_normal_result_unchanged(tool_request):
    middleware = ToolRetryMiddleware(max_attempts=3, initial_delay=0)
    expected = ToolMessage(content="successful docs", tool_call_id="call-1")
    handler = AsyncMock(return_value=expected)

    result = await middleware.awrap_tool_call(tool_request, handler)

    assert result is expected
    handler.assert_awaited_once_with(tool_request)
