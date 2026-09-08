"""Tests for Pylon tool failure propagation."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import requests
from langgraph.prebuilt.tool_node import ToolCallRequest

from src.middleware.tool_retry_middleware import ToolRetryMiddleware
from src.tools.pylon_tools import (
    PylonAuthError,
    _raise_for_status,
    search_support_articles,
)


def test_search_support_articles_raises_for_unauthorized_response():
    """Unauthorized Pylon responses raise instead of returning tool content."""
    response = MagicMock(status_code=401)
    with patch("src.tools.pylon_tools.requests.get", return_value=response):
        with patch("src.tools.pylon_tools._get_api_key", return_value="fake-key"):
            with patch("src.tools.pylon_tools._get_kb_id", return_value="kb-123"):
                with pytest.raises(PylonAuthError) as context:
                    search_support_articles.invoke({"collections": "all"})

    assert str(context.value) == "Pylon API authentication failed with HTTP 401"


def test_raise_for_status_detects_unauthorized_http_error_response():
    """HTTPError responses with 401 status become diagnosable outages."""
    response = MagicMock(status_code=200)
    unauthorized_response = MagicMock(status_code=403)
    response.raise_for_status.side_effect = requests.HTTPError(
        "forbidden", response=unauthorized_response
    )

    with pytest.raises(PylonAuthError) as context:
        _raise_for_status(response, "https://api.usepylon.com/example")

    assert str(context.value) == "Pylon API authentication failed with HTTP 403"


def test_pylon_healthcheck_requests_collections():
    """The Pylon healthcheck performs one authenticated collections request."""
    response = MagicMock(status_code=200)
    with patch("src.tools.pylon_tools.requests.get", return_value=response) as mock_get:
        with patch("src.tools.pylon_tools._get_api_key", return_value="fake-key"):
            with patch("src.tools.pylon_tools._get_kb_id", return_value="kb-123"):
                from src.tools.pylon_tools import pylon_healthcheck

                assert pylon_healthcheck() is True

    mock_get.assert_called_once_with(
        "https://api.usepylon.com/knowledge-bases/kb-123/collections",
        headers={"Authorization": "Bearer fake-key", "Accept": "application/json"},
    )


def test_tool_retry_middleware_propagates_pylon_failures():
    """Pylon outages are marked as tool errors instead of success content."""
    request = ToolCallRequest(
        tool_call={"name": "search_support_articles", "id": "call-1"},
        tool=None,
        state=None,
        runtime=None,
    )
    handler = AsyncMock(side_effect=PylonAuthError("unauthorized"))

    async def invoke():
        return await ToolRetryMiddleware(max_attempts=3).awrap_tool_call(
            request, handler
        )

    result = asyncio.run(invoke())

    assert result.status == "error"
    assert result.content == (
        "The support knowledge base is currently unavailable; answer from "
        "documentation and tell the user this source could not be checked."
    )
    handler.assert_awaited_once()
