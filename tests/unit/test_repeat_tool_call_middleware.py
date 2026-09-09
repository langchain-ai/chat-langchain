"""Tests for per-turn duplicate tool-call suppression."""

import asyncio
from unittest.mock import AsyncMock

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from src.middleware.repeat_tool_call_middleware import RepeatToolCallGuardMiddleware


def _request(turn: str, urls: list[str], call_id: str) -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={
            "name": "check_links",
            "args": {"urls": urls},
            "id": call_id,
            "type": "tool_call",
        },
        tool=None,
        state={"messages": [HumanMessage(content=turn)]},
        runtime=None,
    )


def test_identical_tool_calls_are_invoked_once_per_turn():
    middleware = RepeatToolCallGuardMiddleware()
    handler = AsyncMock(
        return_value=ToolMessage(
            content="Valid links:\n- https://example.com",
            name="check_links",
            tool_call_id="first",
        )
    )

    async def invoke():
        results = []
        for call_id in ("first", "second", "third"):
            results.append(
                await middleware.awrap_tool_call(
                    _request("turn one", ["https://example.com"], call_id),
                    handler,
                )
            )
        return results

    results = asyncio.run(invoke())

    handler.assert_awaited_once()
    assert results[1].content.startswith("NOTE: this exact tool call was already made")
    assert "Valid links:" in results[1].content
    assert "Valid links:" in results[2].content


def test_different_tool_call_arguments_are_not_suppressed():
    middleware = RepeatToolCallGuardMiddleware()
    handler = AsyncMock(
        side_effect=[
            ToolMessage(content="first", name="check_links", tool_call_id="first"),
            ToolMessage(content="second", name="check_links", tool_call_id="second"),
        ]
    )

    async def invoke():
        first = await middleware.awrap_tool_call(
            _request("turn two", ["https://example.com/one"], "first"), handler
        )
        second = await middleware.awrap_tool_call(
            _request("turn two", ["https://example.com/two"], "second"), handler
        )
        return first, second

    first, second = asyncio.run(invoke())

    assert handler.await_count == 2
    assert first.content == "first"
    assert second.content == "second"


def test_memo_resets_for_a_new_turn():
    middleware = RepeatToolCallGuardMiddleware()
    handler = AsyncMock(
        side_effect=[
            ToolMessage(content="turn one", name="check_links", tool_call_id="one"),
            ToolMessage(content="turn two", name="check_links", tool_call_id="two"),
        ]
    )

    async def invoke():
        first = await middleware.awrap_tool_call(
            _request("turn one", ["https://example.com"], "one"), handler
        )
        second = await middleware.awrap_tool_call(
            _request("turn two", ["https://example.com"], "two"), handler
        )
        return first, second

    first, second = asyncio.run(invoke())

    assert handler.await_count == 2
    assert first.content == "turn one"
    assert second.content == "turn two"
