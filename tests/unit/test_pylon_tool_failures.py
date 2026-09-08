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
    search_support_articles,
)


def test_search_support_articles_raises_neutral_error_for_unauthorized_response(caplog):
    """Unauthorized Pylon responses hide operator details from the model."""
    response = MagicMock(status_code=401)
    with patch("src.tools.pylon_tools.requests.get", return_value=response):
        with patch("src.tools.pylon_tools._get_api_key", return_value="fake-key"):
            with patch("src.tools.pylon_tools._get_kb_id", return_value="kb-123"):
                with pytest.raises(PylonUnavailableError) as context:
                    search_support_articles.invoke({"collections": "all"})

    assert (
        str(context.value) == "The support knowledge base is temporarily unavailable."
    )
    assert context.value.status_code == 401
    assert "PYLON_API_KEY" in caplog.text
    assert "kb-123" in caplog.text


def test_raise_for_status_detects_unauthorized_http_error_response(caplog):
    """HTTPError responses retain status while logging operator details."""
    response = MagicMock(status_code=200)
    unauthorized_response = MagicMock(status_code=403)
    response.raise_for_status.side_effect = requests.HTTPError(
        "forbidden", response=unauthorized_response
    )

    with pytest.raises(PylonUnavailableError) as context:
        _raise_for_status(response, "https://api.usepylon.com/example")

    assert (
        str(context.value) == "The support knowledge base is temporarily unavailable."
    )
    assert context.value.status_code == 403
    assert "http_status=403" in caplog.text
    assert "https://api.usepylon.com/example" in caplog.text


def test_tool_retry_middleware_propagates_pylon_failures(caplog):
    """Pylon outages are marked as tool errors instead of success content."""
    request = ToolCallRequest(
        tool_call={"name": "search_support_articles", "id": "call-1"},
        tool=None,
        state=None,
        runtime=None,
    )
    handler = AsyncMock(side_effect=PylonUnavailableError(status_code=401))

    async def invoke():
        return await ToolRetryMiddleware(max_attempts=3).awrap_tool_call(
            request, handler
        )

    result = asyncio.run(invoke())

    assert result.status == "error"
    assert result.content == "The support knowledge base is temporarily unavailable."
    assert "Pylon authorization failure signal" in caplog.text
    handler.assert_awaited_once()
