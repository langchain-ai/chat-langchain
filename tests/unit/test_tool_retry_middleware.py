"""Tests for returned tool failure handling."""

import asyncio
from types import SimpleNamespace

from langchain_core.messages import ToolMessage

from src.middleware.tool_retry_middleware import ToolRetryMiddleware


def _request():
    return SimpleNamespace(
        tool_call={"id": "call-1", "name": "search_docs"},
    )


def test_returned_search_failure_retries_and_normalizes(monkeypatch):
    middleware = ToolRetryMiddleware(max_attempts=2, initial_delay=0)
    results = [
        ToolMessage(
            content="Search failed: 504 Gateway Time-out", tool_call_id="call-1"
        ),
        ToolMessage(
            content="Search failed: 504 Gateway Time-out", tool_call_id="call-1"
        ),
    ]

    async def handler(request):  # noqa: ARG001
        return results.pop(0)

    sleeps = []

    async def sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", sleep)
    result = asyncio.run(middleware.awrap_tool_call(_request(), handler))

    assert len(results) == 0
    assert sleeps == [0]
    assert '"error": "Tool unavailable"' in result.content


def test_returned_filesystem_failure_retries_and_normalizes(monkeypatch):
    middleware = ToolRetryMiddleware(max_attempts=2, initial_delay=0)
    results = [
        ToolMessage(
            content="Docs filesystem query failed: 504 Gateway Time-out",
            tool_call_id="call-1",
        ),
        ToolMessage(
            content="Docs filesystem query failed: 504 Gateway Time-out",
            tool_call_id="call-1",
        ),
    ]

    async def handler(request):  # noqa: ARG001
        return results.pop(0)

    async def sleep(delay):  # noqa: ARG001
        return None

    monkeypatch.setattr(asyncio, "sleep", sleep)
    result = asyncio.run(middleware.awrap_tool_call(_request(), handler))

    assert '"error": "Tool unavailable"' in result.content


def test_successful_docs_page_containing_timeout_is_not_retried(monkeypatch):
    middleware = ToolRetryMiddleware(max_attempts=2, initial_delay=0)
    result = ToolMessage(
        content="This page explains timeout configuration for clients.",
        tool_call_id="call-1",
    )

    async def handler(request):  # noqa: ARG001
        return result

    async def sleep(delay):  # noqa: ARG001
        return None

    monkeypatch.setattr(asyncio, "sleep", sleep)
    assert asyncio.run(middleware.awrap_tool_call(_request(), handler)) is result


def test_exception_path_still_retries_and_normalizes(monkeypatch):
    middleware = ToolRetryMiddleware(max_attempts=2, initial_delay=0)
    calls = 0

    async def handler(request):  # noqa: ARG001
        nonlocal calls
        calls += 1
        raise RuntimeError("HTTP 504 Gateway Time-out")

    async def sleep(delay):  # noqa: ARG001
        return None

    monkeypatch.setattr(asyncio, "sleep", sleep)
    result = asyncio.run(middleware.awrap_tool_call(_request(), handler))

    assert calls == 2
    assert '"error": "Tool unavailable"' in result.content
