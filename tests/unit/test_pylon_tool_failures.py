"""Tests for Pylon tool failure propagation."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import requests
from langgraph.prebuilt.tool_node import ToolCallRequest

from src.middleware.tool_retry_middleware import (
    ToolRetryMiddleware,
    _sanitize_pylon_error,
)
from src.tools.pylon_tools import (
    PylonUnavailableError,
    _raise_for_status,
    check_pylon_credentials,
    search_support_articles,
)


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


def test_raise_for_status_detects_unauthorized_http_error_response():
    """HTTPError responses with 401 status become diagnosable outages."""
    response = MagicMock(status_code=200)
    unauthorized_response = MagicMock(status_code=403)
    response.raise_for_status.side_effect = requests.HTTPError(
        "forbidden", response=unauthorized_response
    )

    with pytest.raises(PylonUnavailableError) as context:
        _raise_for_status(response, "https://api.usepylon.com/example")

    assert "HTTP 403" in str(context.value)
    assert "PYLON_API_KEY" in str(context.value)
    assert "https://api.usepylon.com/example" in str(context.value)


def test_check_pylon_credentials_returns_status_without_raising():
    """Credential checks expose the endpoint status for deployment health checks."""
    response = MagicMock(ok=False, status_code=401)
    with patch("src.tools.pylon_tools.requests.get", return_value=response):
        with patch("src.tools.pylon_tools._get_api_key", return_value="fake-key"):
            with patch("src.tools.pylon_tools._get_kb_id", return_value="kb-123"):
                result = check_pylon_credentials()

    assert result == (False, 401)


def test_tool_retry_middleware_propagates_pylon_failures():
    """Pylon outages return a sanitized structured tool error."""
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
    assert json.loads(result.content)["error"] == "support_kb_unavailable"
    handler.assert_awaited_once()


def test_sanitize_pylon_error_removes_sensitive_details():
    """Pylon diagnostics do not expose infrastructure or credential names."""
    message = (
        "Pylon API returned HTTP 401 for https://api.usepylon.com/knowledge-bases/"
        "123e4567-e89b-12d3-a456-426614174000; rotate PYLON_API_KEY"
    )

    sanitized = _sanitize_pylon_error(message)

    assert "api.usepylon.com" not in sanitized
    assert "123e4567-e89b-12d3-a456-426614174000" not in sanitized
    assert "PYLON_API_KEY" not in sanitized
