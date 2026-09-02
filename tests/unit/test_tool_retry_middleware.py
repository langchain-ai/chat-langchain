"""Tests for tool result and exception retry handling."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage

from src.middleware.tool_retry_middleware import ToolRetryMiddleware


def _request():
    return SimpleNamespace(
        tool_call={"id": "call-1", "name": "search_docs_by_lang_chain"}
    )


@pytest.mark.asyncio
async def test_retries_mcp_failure_result_and_returns_unavailable_payload(monkeypatch):
    middleware = ToolRetryMiddleware(max_attempts=3, initial_delay=0)
    attempts = 0

    async def handler(_request):
        nonlocal attempts
        attempts += 1
        return ToolMessage(
            content=[{"type": "text", "text": "Search failed: 504 Gateway Time-out"}],
            name="search_docs_by_lang_chain",
            tool_call_id="call-1",
        )

    monkeypatch.setattr(
        "src.middleware.tool_retry_middleware.asyncio.sleep", lambda _: _async_noop()
    )
    result = await middleware.awrap_tool_call(_request(), handler)

    assert attempts == 3
    assert json.loads(result.content)["error"] == "Tool unavailable"


async def _async_noop():
    return None


@pytest.mark.asyncio
async def test_normal_docs_page_containing_timeout_passes_through():
    middleware = ToolRetryMiddleware(max_attempts=3)
    result = ToolMessage(
        content="This documentation page explains timeout configuration.",
        name="query_docs_filesystem_docs_by_lang_chain",
        tool_call_id="call-1",
    )
    attempts = 0

    async def handler(_request):
        nonlocal attempts
        attempts += 1
        return result

    assert await middleware.awrap_tool_call(_request(), handler) is result
    assert attempts == 1


@pytest.mark.asyncio
async def test_retries_raised_exception_as_before(monkeypatch):
    middleware = ToolRetryMiddleware(max_attempts=3, initial_delay=0)
    attempts = 0

    async def handler(_request):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("504 Gateway Time-out")

    monkeypatch.setattr(
        "src.middleware.tool_retry_middleware.asyncio.sleep", lambda _: _async_noop()
    )
    result = await middleware.awrap_tool_call(_request(), handler)

    assert attempts == 3
    assert json.loads(result.content)["error"] == "Tool unavailable"
