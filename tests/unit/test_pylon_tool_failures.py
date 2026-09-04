"""Tests for Pylon tool failure propagation."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.prebuilt.tool_node import ToolCallRequest

from src.middleware.tool_retry_middleware import ToolRetryMiddleware
from src.tools.pylon_tools import PylonUnavailableError, search_support_articles


def test_search_support_articles_raises_for_unauthorized_response():
    """Unauthorized Pylon responses raise instead of returning tool content."""
    response = MagicMock(status_code=401)
    with patch("src.tools.pylon_tools.requests.get", return_value=response):
        with patch("src.tools.pylon_tools._get_api_key", return_value="fake-key"):
            with patch("src.tools.pylon_tools._get_kb_id", return_value="kb-123"):
                with pytest.raises(PylonUnavailableError) as context:
                    search_support_articles.invoke({"collections": "all"})

    assert "PYLON_API_KEY" in str(context.value)
    assert "api.usepylon.com" in str(context.value)


def test_tool_retry_middleware_propagates_pylon_failures():
    """Pylon outages are marked as tool errors instead of success content."""
    request = ToolCallRequest(
        tool_call={"name": "search_support_articles", "id": "call-1"},
        tool=None,
        state=None,
        runtime=None,
    )
    handler = AsyncMock(side_effect=PylonUnavailableError("unauthorized"))

    async def invoke():
        return await ToolRetryMiddleware(max_attempts=3).awrap_tool_call(
            request, handler
        )

    result = asyncio.run(invoke())

    assert result.status == "error"
    assert result.content == "unauthorized"
    handler.assert_awaited_once()
