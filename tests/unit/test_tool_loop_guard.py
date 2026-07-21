"""Tests for ToolLoopGuardMiddleware dedup + NOT_FOUND behavior."""

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from src.middleware.tool_loop_guard_middleware import ToolLoopGuardMiddleware


class _FakeRequest:
    def __init__(self, name, args, messages=None, call_id="call-1"):
        self.tool_call = {"name": name, "args": args, "id": call_id}
        self.state = {"messages": messages or []}
        self.runtime = None


def _ai_call(name, args, call_id="prev-1"):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


@pytest.mark.asyncio
async def test_duplicate_identical_call_returns_marker():
    mw = ToolLoopGuardMiddleware()
    args = {"command": "grep EnsembleRetriever /docs"}
    prior = _ai_call("query_docs_filesystem_docs_by_lang_chain", args)
    request = _FakeRequest(
        "query_docs_filesystem_docs_by_lang_chain", args, messages=[prior]
    )

    async def handler(_req):  # pragma: no cover - should be short-circuited
        raise AssertionError("handler should not be called for a duplicate")

    result = await mw.awrap_tool_call(request, handler)

    assert isinstance(result, ToolMessage)
    assert result.content.startswith("DUPLICATE_CALL:")
    assert "query_docs_filesystem_docs_by_lang_chain" in result.content


@pytest.mark.asyncio
async def test_non_duplicate_call_passes_through():
    mw = ToolLoopGuardMiddleware()
    prior = _ai_call(
        "query_docs_filesystem_docs_by_lang_chain", {"command": "grep A /docs"}
    )
    request = _FakeRequest(
        "query_docs_filesystem_docs_by_lang_chain",
        {"command": "grep B /docs"},
        messages=[prior],
    )

    async def handler(req):
        return ToolMessage(
            content="found content",
            name=req.tool_call["name"],
            tool_call_id=req.tool_call["id"],
        )

    result = await mw.awrap_tool_call(request, handler)

    assert isinstance(result, ToolMessage)
    assert result.content == "found content"


@pytest.mark.asyncio
async def test_empty_stdout_returns_not_found():
    mw = ToolLoopGuardMiddleware()
    request = _FakeRequest(
        "query_docs_filesystem_docs_by_lang_chain", {"command": "grep Missing /docs"}
    )

    async def handler(req):
        return ToolMessage(
            content="   ",
            name=req.tool_call["name"],
            tool_call_id=req.tool_call["id"],
        )

    result = await mw.awrap_tool_call(request, handler)

    assert isinstance(result, ToolMessage)
    assert result.content.startswith("NOT_FOUND:")
    assert "grep Missing /docs" in result.content


@pytest.mark.asyncio
async def test_nonzero_exit_returns_not_found():
    mw = ToolLoopGuardMiddleware()
    request = _FakeRequest(
        "query_docs_filesystem_docs_by_lang_chain", {"command": "grep Missing /docs"}
    )

    async def handler(req):
        return ToolMessage(
            content="exit: 1",
            name=req.tool_call["name"],
            tool_call_id=req.tool_call["id"],
        )

    result = await mw.awrap_tool_call(request, handler)

    assert isinstance(result, ToolMessage)
    assert result.content.startswith("NOT_FOUND:")


@pytest.mark.asyncio
async def test_successful_result_preserved():
    mw = ToolLoopGuardMiddleware()
    request = _FakeRequest(
        "query_docs_filesystem_docs_by_lang_chain", {"command": "grep Found /docs"}
    )

    async def handler(req):
        return ToolMessage(
            content="page body with real content",
            name=req.tool_call["name"],
            tool_call_id=req.tool_call["id"],
        )

    result = await mw.awrap_tool_call(request, handler)

    assert result.content == "page body with real content"


def test_sync_duplicate_call_returns_marker():
    mw = ToolLoopGuardMiddleware()
    args = {"query": "ensemble"}
    prior = _ai_call("search_docs_by_lang_chain", args)
    request = _FakeRequest("search_docs_by_lang_chain", args, messages=[prior])

    def handler(_req):  # pragma: no cover - should be short-circuited
        raise AssertionError("handler should not be called for a duplicate")

    result = mw.wrap_tool_call(request, handler)

    assert isinstance(result, ToolMessage)
    assert result.content.startswith("DUPLICATE_CALL:")


def test_unguarded_tool_not_deduped():
    mw = ToolLoopGuardMiddleware()
    args = {"query": "x"}
    prior = _ai_call("search_support_articles", args)
    request = _FakeRequest("search_support_articles", args, messages=[prior])

    def handler(req):
        return ToolMessage(
            content="ok",
            name=req.tool_call["name"],
            tool_call_id=req.tool_call["id"],
        )

    result = mw.wrap_tool_call(request, handler)

    assert result.content == "ok"
