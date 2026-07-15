"""Tests for search-backend outage suppression in ToolRetryMiddleware."""

from __future__ import annotations

import asyncio
import json

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import ToolException
from langgraph.prebuilt.tool_node import ToolCallRequest

from src.middleware.tool_retry_middleware import ToolRetryMiddleware

TOOL_NAME = "search_docs_by_lang_chain"


def _request(call_id: str, turn_message_id: str = "ai-1") -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"name": TOOL_NAME, "args": {"query": "how to stream"}, "id": call_id},
        tool=None,
        state={"messages": [AIMessage(content="", id=turn_message_id)]},
        runtime=None,
    )


def test_outage_signature_suppresses_second_call_same_turn():
    middleware = ToolRetryMiddleware(max_attempts=3, initial_delay=0)
    calls: list[str] = []

    async def handler(request: ToolCallRequest) -> ToolMessage:
        calls.append(request.tool_call["id"])
        return ToolMessage(
            content=json.dumps(
                {"error": "Tool unavailable", "message": "failed after 3 attempts"}
            ),
            name=TOOL_NAME,
            tool_call_id=request.tool_call["id"],
        )

    async def run():
        first = await middleware.awrap_tool_call(_request("c1"), handler)
        second = await middleware.awrap_tool_call(_request("c2"), handler)
        return first, second

    first, second = asyncio.run(run())

    assert calls == ["c1"]
    assert isinstance(first, ToolMessage)
    payload = json.loads(second.content)
    assert payload["error"] == "Tool unavailable"


def test_tool_exception_signature_is_not_retried():
    middleware = ToolRetryMiddleware(max_attempts=3, initial_delay=0)
    calls: list[str] = []

    async def handler(request: ToolCallRequest) -> ToolMessage:
        calls.append(request.tool_call["id"])
        raise ToolException("Search failed: Search failed")

    async def run():
        result = await middleware.awrap_tool_call(_request("c1"), handler)
        await middleware.awrap_tool_call(_request("c2"), handler)
        return result

    result = asyncio.run(run())

    assert calls == ["c1"]
    assert isinstance(result, ToolMessage)


def test_new_turn_resets_outage_flag():
    middleware = ToolRetryMiddleware(max_attempts=3, initial_delay=0)
    calls: list[str] = []

    async def handler(request: ToolCallRequest) -> ToolMessage:
        calls.append(request.tool_call["id"])
        return ToolMessage(
            content=json.dumps({"error": "Tool unavailable", "message": "failed"}),
            name=TOOL_NAME,
            tool_call_id=request.tool_call["id"],
        )

    async def run():
        await middleware.awrap_tool_call(
            _request("c1", turn_message_id="ai-1"), handler
        )
        await middleware.awrap_tool_call(
            _request("c2", turn_message_id="ai-2"), handler
        )

    asyncio.run(run())

    assert calls == ["c1", "c2"]
