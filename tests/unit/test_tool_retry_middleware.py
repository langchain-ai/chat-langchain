"""Tests for docs tool failures and bounded tool retries."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from src.middleware import tool_retry_middleware as retry_module
from src.middleware.tool_retry_middleware import ToolRetryMiddleware

DOCS_TOOLS = (
    "search_docs_by_lang_chain",
    "query_docs_filesystem_docs_by_lang_chain",
)


def _request(tool_name=DOCS_TOOLS[0]):
    return SimpleNamespace(tool_call={"name": tool_name, "id": "call-1"})


def _message(content):
    return ToolMessage(content=content, tool_call_id="call-1")


def _assert_unavailable(result, request):
    payload = json.loads(result.content)
    assert payload["error"] == "Tool unavailable"
    assert payload["tool"] == request.tool_call["name"]
    assert result.name == request.tool_call["name"]
    assert result.tool_call_id == request.tool_call["id"]
    assert "documentation could not be found" in payload["suggestion"]
    if request.tool_call["name"] == DOCS_TOOLS[0]:
        assert DOCS_TOOLS[1] in payload["suggestion"]
    return payload


@pytest.mark.parametrize(
    "content",
    [
        "Search failed: 504 Gateway Time-out",
        "Docs filesystem query failed: 504 Gateway Time-out",
        [{"type": "text", "text": "Search failed: 504 Gateway Time-out"}],
        ["Search failed: 504 Gateway Time-out"],
        [{"type": "image", "url": "image"}, {"type": "text", "text": "timeout"}],
        "Service unavailable",
        "Search failed: unexpected upstream response",
        "Search failed: " + "upstream error " * 30,
        "timeout" + " " * 291 + "!",
    ],
)
@pytest.mark.parametrize("tool_name", DOCS_TOOLS)
def test_content_failure_is_normalized_when_exhausted(content, tool_name):
    request = _request(tool_name)
    handler = AsyncMock(return_value=_message(content))
    middleware = ToolRetryMiddleware(max_attempts=1)

    result = asyncio.run(middleware.awrap_tool_call(request, handler))

    _assert_unavailable(result, request)
    assert handler.await_count == 1


def test_fast_content_failure_retries_with_existing_backoff(monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr(retry_module.asyncio, "sleep", sleep)
    success = _message("Documentation content")
    handler = AsyncMock(
        side_effect=[
            _message("Search failed: 504 Gateway Time-out"),
            _message("Search failed: unexpected upstream response"),
            success,
        ]
    )
    middleware = ToolRetryMiddleware(initial_delay=0.25, backoff_factor=3.0)

    result = asyncio.run(middleware.awrap_tool_call(_request(), handler))

    assert result is success
    assert handler.await_count == 3
    assert [call.args[0] for call in sleep.await_args_list] == [0.25, 0.75]


@pytest.mark.parametrize("elapsed", [5.0, 10.0])
def test_slow_content_failure_is_not_retried(monkeypatch, elapsed):
    times = iter([100.0, 100.0 + elapsed])
    monkeypatch.setattr(retry_module.time, "perf_counter", lambda: next(times))
    request = _request()
    handler = AsyncMock(return_value=_message("Search failed: 504 Gateway Time-out"))

    result = asyncio.run(ToolRetryMiddleware().awrap_tool_call(request, handler))

    _assert_unavailable(result, request)
    assert handler.await_count == 1


def test_repeated_content_failure_exhausts_attempts(monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr(retry_module.asyncio, "sleep", sleep)
    request = _request()
    handler = AsyncMock(return_value=_message("Search failed: 504 Gateway Time-out"))

    result = asyncio.run(ToolRetryMiddleware().awrap_tool_call(request, handler))

    _assert_unavailable(result, request)
    assert handler.await_count == 3
    assert sleep.await_count == 2


@pytest.mark.parametrize("tool_name", DOCS_TOOLS)
def test_docs_timeout_is_normalized_without_retry(tool_name):
    request = _request(tool_name)
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def slow_handler(request):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    handler = AsyncMock(side_effect=slow_handler)
    middleware = ToolRetryMiddleware(tool_timeouts={tool_name: 0.01})

    async def run_call():
        return await asyncio.wait_for(middleware.awrap_tool_call(request, handler), 1.0)

    result = asyncio.run(run_call())

    payload = _assert_unavailable(result, request)
    assert "timed out after 0.01 seconds" in payload["details"]
    assert handler.await_count == 1
    assert started.is_set()
    assert cancelled.is_set()


def test_default_docs_timeouts_are_independent():
    middleware = ToolRetryMiddleware()
    assert middleware.tool_timeouts == dict.fromkeys(DOCS_TOOLS, 20.0)
    middleware.tool_timeouts[DOCS_TOOLS[0]] = 1.0
    assert ToolRetryMiddleware().tool_timeouts[DOCS_TOOLS[0]] == 20.0


@pytest.mark.parametrize(
    "result",
    [
        _message("Documentation content"),
        _message("Configure the timeout setting. " * 20),
        _message("timeout" + " " * 292 + "!"),
        Command(update={"messages": [_message("Search failed: 504 Gateway Time-out")]}),
    ],
)
def test_successful_docs_results_are_unchanged(result):
    handler = AsyncMock(return_value=result)

    actual = asyncio.run(ToolRetryMiddleware().awrap_tool_call(_request(), handler))

    assert actual is result
    assert handler.await_count == 1


def test_non_docs_content_is_not_inspected_or_timed_out(monkeypatch):
    wait_for = AsyncMock(side_effect=AssertionError("Unexpected timeout wrapper"))
    monkeypatch.setattr(retry_module.asyncio, "wait_for", wait_for)
    result = _message("Search failed: 504 Gateway Time-out")
    handler = AsyncMock(return_value=result)

    actual = asyncio.run(
        ToolRetryMiddleware().awrap_tool_call(_request("other_tool"), handler)
    )

    assert actual is result
    assert handler.await_count == 1
    wait_for.assert_not_awaited()


@pytest.mark.parametrize(
    "error", [RuntimeError("HTTP 504"), asyncio.TimeoutError("timeout")]
)
def test_non_docs_retryable_exceptions_still_retry(monkeypatch, error):
    sleep = AsyncMock()
    monkeypatch.setattr(retry_module.asyncio, "sleep", sleep)
    success = _message("Recovered")
    handler = AsyncMock(side_effect=[error, success])

    result = asyncio.run(
        ToolRetryMiddleware().awrap_tool_call(_request("other_tool"), handler)
    )

    assert result is success
    assert handler.await_count == 2
    sleep.assert_awaited_once_with(0.5)


def test_non_docs_no_results_exception_is_unchanged():
    handler = AsyncMock(side_effect=RuntimeError("No results found"))

    result = asyncio.run(
        ToolRetryMiddleware().awrap_tool_call(_request("other_tool"), handler)
    )

    assert result.content == "No results found."
    assert handler.await_count == 1
