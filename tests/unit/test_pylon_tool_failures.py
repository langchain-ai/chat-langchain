"""Tests for Pylon tool failure propagation."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import requests
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

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


def test_tool_retry_middleware_recovers_malformed_tool_name():
    """Malformed names resolve to the registered tool before execution."""
    from langchain_core.tools import StructuredTool
    from langgraph.prebuilt.tool_node import ToolRuntime

    tool = StructuredTool.from_function(
        lambda query: query,
        name="search_support_articles",
        description="Search support articles.",
    )
    request = ToolCallRequest(
        tool_call={
            "name": "reasoning\n<ctrl>:call:default_api:search_support_articles",
            "args": {"query": "docs"},
            "id": "call-2",
        },
        tool=None,
        state=None,
        runtime=ToolRuntime(
            state=None,
            context=None,
            config={},
            stream_writer=lambda _: None,
            tool_call_id="call-2",
            store=None,
            tools=[tool],
        ),
    )
    handler = AsyncMock(return_value=ToolMessage(content="ok", tool_call_id="call-2"))

    async def invoke():
        return await ToolRetryMiddleware().awrap_tool_call(request, handler)

    result = asyncio.run(invoke())

    assert result.content == "ok"
    handler.assert_awaited_once()
    assert handler.await_args.args[0].tool_call["name"] == "search_support_articles"
    assert handler.await_args.args[0].tool_call["args"] == {"query": "docs"}


def test_tool_retry_middleware_prunes_invalid_tool_history():
    """Invalid calls and paired tool messages are removed before the model call."""
    from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
    from langchain_core.tools import StructuredTool

    tool = StructuredTool.from_function(
        lambda query: query,
        name="valid",
        description="A valid test tool.",
    )
    middleware = ToolRetryMiddleware()
    middleware._set_tool_registry([tool])
    ai_message = AIMessage(
        content="",
        id="ai-1",
        tool_calls=[
            {"name": "valid", "args": {}, "id": "valid-call", "type": "tool_call"},
            {"name": "invalid", "args": {}, "id": "invalid-call", "type": "tool_call"},
        ],
    )
    state = {
        "messages": [
            HumanMessage(content="search", id="human-1"),
            ai_message,
            ToolMessage(content="valid result", tool_call_id="valid-call", id="tool-1"),
            ToolMessage(content="rejected", tool_call_id="invalid-call", id="tool-2"),
        ]
    }

    result = middleware.before_model(state, None)

    assert result is not None
    assert isinstance(result["messages"][0], RemoveMessage)
    assert isinstance(result["messages"][1], RemoveMessage)
    assert result["messages"][1].id == "tool-2"
    assert result["messages"][-1].tool_calls == [ai_message.tool_calls[0]]
