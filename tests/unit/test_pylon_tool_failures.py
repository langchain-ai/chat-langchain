"""Tests for Pylon tool failure propagation."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import requests
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from src.middleware.support_disclosure_middleware import SupportDisclosureMiddleware
from src.middleware.tool_retry_middleware import ToolRetryMiddleware
from src.tools.pylon_tools import (
    PylonUnavailableError,
    _raise_for_status,
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


def test_support_outage_disclosure_is_added_after_retry():
    middleware = SupportDisclosureMiddleware()
    calls: list[ModelRequest] = []
    request = ModelRequest(
        model=object(),
        messages=[
            HumanMessage(content="How do I update my billing details?"),
            ToolMessage(
                content="PylonUnavailableError: Pylon API returned HTTP 401",
                name="search_support_articles",
                status="error",
                tool_call_id="search",
            ),
        ],
    )

    async def handler(request: ModelRequest) -> ModelResponse:
        calls.append(request)
        return ModelResponse(
            result=[AIMessage(content="Use the billing settings page.")]
        )

    response = asyncio.run(middleware.awrap_model_call(request, handler))

    assert len(calls) == 2
    assert response.result[0].content.endswith(
        "Support articles could not be consulted, so this answer is based on official documentation only."
    )


def test_existing_support_outage_disclosure_is_unchanged():
    middleware = SupportDisclosureMiddleware()
    content = (
        "Support articles could not be consulted; based on official documentation only."
    )
    request = ModelRequest(
        model=object(),
        messages=[
            HumanMessage(content="How do I update my billing details?"),
            ToolMessage(
                content="PylonUnavailableError",
                name="get_support_article_content",
                status="error",
                tool_call_id="content",
            ),
        ],
    )

    async def handler(request: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content=content)])

    response = asyncio.run(middleware.awrap_model_call(request, handler))

    assert response.result[0].content == content


def test_successful_support_tool_does_not_change_answer():
    middleware = SupportDisclosureMiddleware()
    content = "Use the billing settings page."
    request = ModelRequest(
        model=object(),
        messages=[
            HumanMessage(content="How do I update my billing details?"),
            ToolMessage(
                content="Article content",
                name="search_support_articles",
                status="success",
                tool_call_id="search",
            ),
        ],
    )

    async def handler(request: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content=content)])

    response = asyncio.run(middleware.awrap_model_call(request, handler))

    assert response.result[0].content == content
