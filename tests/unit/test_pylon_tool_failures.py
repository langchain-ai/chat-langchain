"""Tests for Pylon tool failure propagation."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import requests
from langgraph.prebuilt.tool_node import ToolCallRequest

from src.middleware.tool_retry_middleware import ToolRetryMiddleware
from src.tools.pylon_tools import (
    PylonUnavailableError,
    _raise_for_status,
    probe_pylon_health,
    search_support_articles,
)


def test_search_support_articles_raises_safe_error_for_unauthorized_response(caplog):
    """Unauthorized Pylon responses do not expose operator remediation."""
    response = MagicMock(status_code=401)
    with patch("src.tools.pylon_tools.requests.get", return_value=response):
        with patch("src.tools.pylon_tools._get_api_key", return_value="fake-key"):
            with patch("src.tools.pylon_tools._get_kb_id", return_value="kb-123"):
                with pytest.raises(PylonUnavailableError) as context:
                    search_support_articles.invoke({"collections": "all"})

    assert "Support knowledge base is temporarily unavailable" in str(context.value)
    assert "PYLON_API_KEY" not in str(context.value)
    assert "api.usepylon.com" not in str(context.value)
    assert "PYLON_API_KEY" in caplog.text
    assert "HTTP 401" in caplog.text


def test_raise_for_status_detects_unauthorized_http_error_response():
    """HTTPError responses with 401 status become diagnosable outages."""
    response = MagicMock(status_code=200)
    unauthorized_response = MagicMock(status_code=403)
    response.raise_for_status.side_effect = requests.HTTPError(
        "forbidden", response=unauthorized_response
    )

    with pytest.raises(PylonUnavailableError) as context:
        _raise_for_status(response, "https://api.usepylon.com/example")

    assert "Support knowledge base is temporarily unavailable" in str(context.value)
    assert "PYLON_API_KEY" not in str(context.value)
    assert "https://api.usepylon.com/example" not in str(context.value)


def test_tool_retry_middleware_returns_structured_pylon_failure():
    """Pylon outages return a structured non-retryable tool error."""
    request = ToolCallRequest(
        tool_call={"name": "search_support_articles", "id": "call-1"},
        tool=None,
        state=None,
        runtime=None,
    )
    handler = AsyncMock(
        side_effect=PylonUnavailableError(
            "rotate PYLON_API_KEY using the operator remediation"
        )
    )

    async def invoke():
        return await ToolRetryMiddleware(max_attempts=3).awrap_tool_call(
            request, handler
        )

    result = asyncio.run(invoke())

    assert result.status == "error"
    assert json.loads(result.content) == {
        "error": "support_kb_unavailable",
        "tool": "search_support_articles",
        "instruction": (
            "Do not retry this tool this turn. Answer from documentation and "
            "explicitly state that the support knowledge base could not be consulted."
        ),
    }
    assert "PYLON_API_KEY" not in result.content
    assert "credential" not in result.content.lower()
    handler.assert_awaited_once()


def test_pylon_health_probe_uses_collections_endpoint():
    """The health probe checks collections without changing tool behavior."""
    response = MagicMock(status_code=200)
    response.json.return_value = {"data": []}
    with patch("src.tools.pylon_tools.requests.get", return_value=response) as request:
        with patch("src.tools.pylon_tools._get_api_key", return_value="fake-key"):
            with patch("src.tools.pylon_tools._get_kb_id", return_value="kb-123"):
                assert probe_pylon_health() is True

    request.assert_called_once_with(
        "https://api.usepylon.com/knowledge-bases/kb-123/collections",
        headers={"Authorization": "Bearer fake-key", "Accept": "application/json"},
        timeout=10,
    )
