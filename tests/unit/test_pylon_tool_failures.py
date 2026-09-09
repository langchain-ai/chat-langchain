"""Tests for Pylon tool failure propagation."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import requests
from langgraph.prebuilt.tool_node import ToolCallRequest

import src.tools.pylon_tools as pylon_tools
from src.middleware.tool_retry_middleware import ToolRetryMiddleware
from src.tools.pylon_tools import (
    PylonUnavailableError,
    _raise_for_status,
    search_support_articles,
)


@pytest.fixture(autouse=True)
def reset_pylon_state():
    pylon_tools._auth_failure_cooldown_until = None
    pylon_tools._articles_cache = None
    pylon_tools._collections_cache = None


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


def test_tool_retry_middleware_propagates_pylon_failures():
    """Pylon outages return a sanitized model-facing error payload."""
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
    assert json.loads(result.content) == {
        "error": "support_kb_unavailable",
        "message": (
            "The support knowledge base could not be reached for this request. "
            "Answer from documentation sources and tell the user the support KB "
            "was not consulted."
        ),
    }
    assert "unauthorized" not in result.content
    handler.assert_awaited_once()


def test_authentication_failure_cooldown_skips_follow_up_request():
    """A 401 response prevents repeated requests during the cooldown."""
    first_response = MagicMock(status_code=401)

    with patch(
        "src.tools.pylon_tools.requests.get", return_value=first_response
    ) as mock_get:
        with patch("src.tools.pylon_tools._get_api_key", return_value="fake-key"):
            with patch("src.tools.pylon_tools._get_kb_id", return_value="kb-123"):
                with pytest.raises(PylonUnavailableError):
                    pylon_tools._fetch_all_articles()
                with pytest.raises(PylonUnavailableError):
                    pylon_tools._fetch_all_articles()

    mock_get.assert_called_once()
