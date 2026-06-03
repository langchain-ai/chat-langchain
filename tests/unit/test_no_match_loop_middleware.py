"""Tests for NoMatchLoopGuardMiddleware exit-loop and empty-content guards."""

import asyncio
import os

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

os.environ["USE_LOCAL_PROMPTS"] = "1"

from src.middleware.no_match_loop_middleware import (
    EMPTY_CONTENT_FALLBACK,
    NoMatchLoopGuardMiddleware,
)

FILESYSTEM_TOOL = "query_docs_filesystem_docs_by_lang_chain"


def _no_match_tool_message(i: int) -> ToolMessage:
    return ToolMessage(
        content=f"exit: 1\nstdout:\nstderr: rg: no matches found (attempt {i})",
        name=FILESYSTEM_TOOL,
        tool_call_id=f"call-{i}",
    )


def _request(state: dict, call_id: str = "next") -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"name": FILESYSTEM_TOOL, "args": {"command": "rg foo"}, "id": call_id},
        tool=None,
        state=state,
        runtime=None,
    )


def test_short_circuits_after_threshold_consecutive_no_match_results():
    middleware = NoMatchLoopGuardMiddleware(max_consecutive_no_match=5)
    messages = [HumanMessage(content="how do I use middleware?")]
    for i in range(30):
        messages.append(
            AIMessage(
                content="",
                tool_calls=[{"name": FILESYSTEM_TOOL, "args": {}, "id": f"call-{i}"}],
            )
        )
        messages.append(_no_match_tool_message(i))

    request = _request({"messages": messages})

    handler_called = {"count": 0}

    async def handler(_req):
        handler_called["count"] += 1
        return ToolMessage(content="exit: 1", name=FILESYSTEM_TOOL, tool_call_id="next")

    result = asyncio.run(middleware.awrap_tool_call(request, handler))

    assert isinstance(result, ToolMessage)
    assert handler_called["count"] == 0
    assert "filesystem search has been disabled" in result.content
    assert "search_docs_by_lang_chain" in result.content


def test_allows_call_before_threshold_reached():
    middleware = NoMatchLoopGuardMiddleware(max_consecutive_no_match=5)
    messages = [HumanMessage(content="how do I use middleware?")]
    for i in range(3):
        messages.append(
            AIMessage(
                content="",
                tool_calls=[{"name": FILESYSTEM_TOOL, "args": {}, "id": f"call-{i}"}],
            )
        )
        messages.append(_no_match_tool_message(i))

    request = _request({"messages": messages})

    async def handler(_req):
        return ToolMessage(content="success", name=FILESYSTEM_TOOL, tool_call_id="next")

    result = asyncio.run(middleware.awrap_tool_call(request, handler))
    assert isinstance(result, ToolMessage)
    assert result.content == "success"


def test_streak_resets_on_successful_result():
    middleware = NoMatchLoopGuardMiddleware(max_consecutive_no_match=5)
    messages: list = [HumanMessage(content="hi")]
    for i in range(10):
        messages.append(
            AIMessage(
                content="",
                tool_calls=[{"name": FILESYSTEM_TOOL, "args": {}, "id": f"call-{i}"}],
            )
        )
        messages.append(_no_match_tool_message(i))
    messages.append(
        AIMessage(
            content="",
            tool_calls=[{"name": FILESYSTEM_TOOL, "args": {}, "id": "good"}],
        )
    )
    messages.append(
        ToolMessage(
            content="Found docs: middleware basics...",
            name=FILESYSTEM_TOOL,
            tool_call_id="good",
        )
    )

    request = _request({"messages": messages})

    async def handler(_req):
        return ToolMessage(content="ok", name=FILESYSTEM_TOOL, tool_call_id="next")

    result = asyncio.run(middleware.awrap_tool_call(request, handler))
    assert result.content == "ok"


def test_after_model_replaces_empty_final_content():
    middleware = NoMatchLoopGuardMiddleware()
    state = {
        "messages": [
            HumanMessage(content="What is langgraph?"),
            AIMessage(id="ai-1", content=""),
        ]
    }
    update = middleware.after_model(state, runtime=None)
    assert update is not None
    assert "messages" in update
    replacement = update["messages"][0]
    assert isinstance(replacement, AIMessage)
    assert replacement.content == EMPTY_CONTENT_FALLBACK
    assert replacement.id == "ai-1"


def test_after_model_skips_when_tool_calls_pending():
    middleware = NoMatchLoopGuardMiddleware()
    state = {
        "messages": [
            HumanMessage(content="hi"),
            AIMessage(
                content="",
                tool_calls=[{"name": FILESYSTEM_TOOL, "args": {}, "id": "x"}],
            ),
        ]
    }
    assert middleware.after_model(state, runtime=None) is None


def test_after_model_skips_when_content_present():
    middleware = NoMatchLoopGuardMiddleware()
    state = {
        "messages": [
            HumanMessage(content="hi"),
            AIMessage(content="Here is the answer."),
        ]
    }
    assert middleware.after_model(state, runtime=None) is None
