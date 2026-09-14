"""Tests for duplicate tool-call suppression."""

import asyncio

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from src.middleware.duplicate_tool_call_middleware import DuplicateToolCallMiddleware


def _request(turn_id: str, call_id: str, args: dict[str, object]) -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"name": "check_links", "id": call_id, "args": args},
        tool=None,
        state={"messages": [HumanMessage(content="Check this link", id=turn_id)]},
        runtime=None,
    )


def test_suppresses_identical_second_call():
    middleware = DuplicateToolCallMiddleware()
    calls: list[dict[str, object]] = []

    async def handler(request: ToolCallRequest) -> ToolMessage:
        calls.append(request.tool_call)
        return ToolMessage(
            content="Valid links:\n- https://example.com",
            name="check_links",
            tool_call_id=request.tool_call["id"],
        )

    async def invoke() -> tuple[ToolMessage, ToolMessage]:
        first = await middleware.awrap_tool_call(
            _request("turn-1", "call-1", {"urls": ["https://example.com"]}), handler
        )
        second = await middleware.awrap_tool_call(
            _request("turn-1", "call-2", {"urls": ["https://example.com"]}), handler
        )
        return first, second

    first, second = asyncio.run(invoke())

    assert len(calls) == 1
    assert first.content == "Valid links:\n- https://example.com"
    assert second.name == "check_links"
    assert second.tool_call_id == "call-2"
    assert second.content.startswith("[duplicate call suppressed]")
    assert "https://example.com" in second.content


def test_executes_distinct_arguments():
    middleware = DuplicateToolCallMiddleware()
    calls: list[dict[str, object]] = []

    async def handler(request: ToolCallRequest) -> ToolMessage:
        calls.append(request.tool_call)
        return ToolMessage(
            content=str(request.tool_call["args"]),
            name="check_links",
            tool_call_id=request.tool_call["id"],
        )

    async def invoke() -> tuple[ToolMessage, ToolMessage]:
        first = await middleware.awrap_tool_call(
            _request("turn-1", "call-1", {"urls": ["https://example.com"]}), handler
        )
        second = await middleware.awrap_tool_call(
            _request("turn-1", "call-2", {"urls": ["https://langchain.com"]}), handler
        )
        return first, second

    first, second = asyncio.run(invoke())

    assert len(calls) == 2
    assert "example.com" in first.content
    assert "langchain.com" in second.content


def test_resets_cache_on_new_human_turn():
    middleware = DuplicateToolCallMiddleware()
    calls: list[str] = []

    async def handler(request: ToolCallRequest) -> ToolMessage:
        calls.append(request.tool_call["id"])
        return ToolMessage(
            content="fresh result",
            name="check_links",
            tool_call_id=request.tool_call["id"],
        )

    async def invoke() -> tuple[ToolMessage, ToolMessage]:
        first = await middleware.awrap_tool_call(
            _request("turn-1", "call-1", {"urls": ["https://example.com"]}), handler
        )
        second = await middleware.awrap_tool_call(
            _request("turn-2", "call-2", {"urls": ["https://example.com"]}), handler
        )
        return first, second

    first, second = asyncio.run(invoke())

    assert calls == ["call-1", "call-2"]
    assert not first.content.startswith("[duplicate call suppressed]")
    assert not second.content.startswith("[duplicate call suppressed]")
