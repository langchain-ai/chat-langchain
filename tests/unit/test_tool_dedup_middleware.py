"""Tests for tool-call deduplication and per-trace cap."""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from src.middleware.tool_dedup_middleware import ToolCallDedupMiddleware


def _request(name, args, call_id, messages):
    return ToolCallRequest(
        tool_call={"name": name, "args": args, "id": call_id},
        tool=None,
        state={"messages": messages},
        runtime=None,
    )


async def _handler_factory(sentinel):
    async def handler(request):
        return ToolMessage(
            content=sentinel,
            name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"],
        )

    return handler


@pytest.mark.asyncio
async def test_first_call_executes_tool():
    middleware = ToolCallDedupMiddleware()
    messages = [HumanMessage(content="hi", id="h1")]
    request = _request("search", {"q": "middleware"}, "c1", messages)
    handler = await _handler_factory("fresh-result")

    result = await middleware.awrap_tool_call(request, handler)

    assert result.content == "fresh-result"


@pytest.mark.asyncio
async def test_second_identical_call_returns_cached_result():
    middleware = ToolCallDedupMiddleware()
    messages = [
        AIMessage(
            content="",
            tool_calls=[{"name": "search", "args": {"q": "middleware"}, "id": "c1"}],
        ),
        ToolMessage(content="cached-result", name="search", tool_call_id="c1"),
    ]
    request = _request("search", {"q": "middleware"}, "c2", messages)

    async def handler(request):  # pragma: no cover - should not run
        raise AssertionError("tool should not be re-executed")

    result = await middleware.awrap_tool_call(request, handler)

    assert result.content == "cached-result"


@pytest.mark.asyncio
async def test_third_identical_call_injects_stop_directive():
    middleware = ToolCallDedupMiddleware()
    messages = [
        AIMessage(
            content="",
            tool_calls=[{"name": "search", "args": {"q": "middleware"}, "id": "c1"}],
        ),
        ToolMessage(content="r1", name="search", tool_call_id="c1"),
        AIMessage(
            content="",
            tool_calls=[{"name": "search", "args": {"q": "middleware"}, "id": "c2"}],
        ),
        ToolMessage(content="r2", name="search", tool_call_id="c2"),
    ]
    request = _request("search", {"q": "middleware"}, "c3", messages)

    async def handler(request):  # pragma: no cover - should not run
        raise AssertionError("tool should not be re-executed")

    result = await middleware.awrap_tool_call(request, handler)

    payload = json.loads(result.content)
    assert payload["notice"] == "Duplicate tool call blocked."
    assert "Do not call it again" in payload["instruction"]


@pytest.mark.asyncio
async def test_cap_forces_final_answer():
    middleware = ToolCallDedupMiddleware(max_tool_calls=2)
    messages = [
        AIMessage(
            content="",
            tool_calls=[{"name": "search", "args": {"q": "a"}, "id": "c1"}],
        ),
        ToolMessage(content="r1", name="search", tool_call_id="c1"),
        AIMessage(
            content="",
            tool_calls=[{"name": "search", "args": {"q": "b"}, "id": "c2"}],
        ),
        ToolMessage(content="r2", name="search", tool_call_id="c2"),
    ]
    request = _request("search", {"q": "c"}, "c3", messages)
    handler = await _handler_factory("should-not-run")

    result = await middleware.awrap_tool_call(request, handler)

    payload = json.loads(result.content)
    assert payload["notice"] == "Tool-call limit reached."
