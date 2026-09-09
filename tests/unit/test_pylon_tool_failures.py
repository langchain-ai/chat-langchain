"""Tests for Pylon tool failure propagation."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import requests
from langgraph.prebuilt.tool_node import ToolCallRequest

from src.middleware.tool_retry_middleware import ToolRetryMiddleware
from src.tools.pylon_tools import (
    PylonUnavailableError,
    _raise_for_status,
    get_support_article_content,
    search_support_articles,
)


def test_unauthorized_response_redacts_operator_details_from_tool_content():
    """Unauthorized Pylon responses expose only the safe model message."""
    import src.tools.pylon_tools as pylon_tools

    pylon_tools._kb_unavailable_until = 0
    response = MagicMock(status_code=401)
    with patch("src.tools.pylon_tools.requests.get", return_value=response):
        with patch("src.tools.pylon_tools._get_api_key", return_value="fake-key"):
            with patch("src.tools.pylon_tools._get_kb_id", return_value="kb-123"):
                with pytest.raises(PylonUnavailableError) as context:
                    search_support_articles.invoke({"collections": "all"})

    assert "usepylon.com" not in str(context.value)
    assert "PYLON_API_KEY" not in str(context.value)
    assert "kb-123" not in str(context.value)
    assert "usepylon.com" in context.value.operator_detail
    assert "PYLON_API_KEY" in context.value.operator_detail
    assert "kb-123" in context.value.operator_detail


def test_raise_for_status_detects_unauthorized_http_error_response():
    """HTTPError responses with 401 status become diagnosable outages."""
    response = MagicMock(status_code=200)
    unauthorized_response = MagicMock(status_code=403)
    response.raise_for_status.side_effect = requests.HTTPError(
        "forbidden", response=unauthorized_response
    )

    with pytest.raises(PylonUnavailableError) as context:
        _raise_for_status(response, "https://api.usepylon.com/example")

    assert "HTTP 403" in context.value.operator_detail
    assert "PYLON_API_KEY" in context.value.operator_detail
    assert "https://api.usepylon.com/example" in context.value.operator_detail
    assert "usepylon.com" not in str(context.value)
    assert "PYLON_API_KEY" not in str(context.value)


def test_breaker_prevents_second_support_kb_request():
    """An authentication failure prevents another KB request during the window."""
    import src.tools.pylon_tools as pylon_tools

    pylon_tools._kb_unavailable_until = 0
    response = MagicMock(status_code=401)
    with patch("src.tools.pylon_tools.requests.get", return_value=response) as request:
        with patch("src.tools.pylon_tools._get_api_key", return_value="fake-key"):
            with patch("src.tools.pylon_tools._get_kb_id", return_value="kb-123"):
                with pytest.raises(PylonUnavailableError):
                    search_support_articles.invoke({"collections": "all"})
                with pytest.raises(PylonUnavailableError):
                    get_support_article_content.invoke({"article_id": "article-1"})

    request.assert_called_once()


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
    assert "The support knowledge base is temporarily unavailable" in result.content
    handler.assert_awaited_once()
