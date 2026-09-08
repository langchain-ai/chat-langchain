"""Tests for Pylon tool failure propagation."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import requests
from langgraph.prebuilt.tool_node import ToolCallRequest

from src.middleware.tool_retry_middleware import ToolRetryMiddleware
from src.tools.pylon_tools import (
    PylonCredentialConfigurationError,
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


def test_tool_retry_middleware_sanitizes_pylon_failures(caplog):
    """Pylon outages are logged privately and sanitized for the model."""
    request = ToolCallRequest(
        tool_call={"name": "search_support_articles", "id": "call-1"},
        tool=None,
        state=None,
        runtime=None,
    )
    operator_error = (
        "Pylon API returned HTTP 401 for "
        "https://api.usepylon.com/knowledge-bases/kb-123/collections; "
        "check or rotate PYLON_API_KEY configuration or credentials."
    )
    handler = AsyncMock(side_effect=PylonUnavailableError(operator_error))

    async def invoke():
        return await ToolRetryMiddleware(max_attempts=3).awrap_tool_call(
            request, handler
        )

    result = asyncio.run(invoke())

    assert result.status == "error"
    payload = json.loads(result.content)
    assert payload == {
        "error": "Support knowledge base unavailable",
        "tool": "search_support_articles",
        "user_notice": "Tell the user that the support knowledge base could not be consulted.",
    }
    assert "https://api.usepylon.com" not in result.content
    assert "kb-123" not in result.content
    assert "PYLON_API_KEY" not in result.content
    assert operator_error in caplog.text
    handler.assert_awaited_once()


def test_check_pylon_credentials_raises_configuration_error_on_unauthorized():
    """Unauthorized credential checks raise a distinct configuration error."""
    response = MagicMock(status_code=401)
    with patch("src.tools.pylon_tools.requests.get", return_value=response) as mock_get:
        with patch("src.tools.pylon_tools._get_api_key", return_value="fake-key"):
            with patch("src.tools.pylon_tools._get_kb_id", return_value="kb-123"):
                with pytest.raises(PylonCredentialConfigurationError) as context:
                    check_pylon_credentials()

    assert "HTTP 401" in str(context.value)
    assert mock_get.call_count == 1
    mock_get.assert_called_once_with(
        "https://api.usepylon.com/knowledge-bases/kb-123/collections",
        headers={"Authorization": "Bearer fake-key", "Accept": "application/json"},
    )
