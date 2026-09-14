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


@pytest.mark.parametrize("article_id", ["0", "N/A"])
def test_get_support_article_content_rejects_placeholder_ids(article_id):
    """Placeholder article IDs return guidance to search first."""
    result = get_support_article_content.invoke({"article_id": article_id})

    assert result == (
        f"Invalid article_id '{article_id}'. Call search_support_articles first "
        "and copy an `id` from its results."
    )


def test_get_support_article_content_rejects_all_zero_uuid():
    """The all-zero UUID is not accepted as an article ID."""
    result = get_support_article_content.invoke(
        {"article_id": "00000000-0000-0000-0000-000000000000"}
    )

    assert result.startswith("Invalid article_id '")
    assert "copy an `id` from its results" in result


def test_get_support_article_content_uses_valid_uuid_lookup():
    """A valid UUID continues through the normal article lookup path."""
    article_id = "123e4567-e89b-12d3-a456-426614174000"
    article = {
        "id": article_id,
        "title": "Test article",
        "identifier": "test",
        "slug": "article",
        "collection_id": "collection-id",
        "current_published_content_html": "<p>Content</p>",
    }

    with patch(
        "src.tools.pylon_tools._fetch_all_articles", return_value=[article]
    ) as fetch_articles:
        with patch(
            "src.tools.pylon_tools._fetch_collections",
            return_value={"General": "collection-id"},
        ) as fetch_collections:
            result = get_support_article_content.invoke({"article_id": article_id})

    fetch_articles.assert_called_once_with()
    fetch_collections.assert_called_once_with()
    assert "Title: Test article" in result
    assert "Content:\n<p>Content</p>" in result


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
