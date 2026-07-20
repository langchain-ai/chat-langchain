"""Tests for ToolRetryMiddleware repeated-identical-call detection.

These tests do NOT require network access or LangSmith credentials.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from src.middleware.tool_retry_middleware import ToolRetryMiddleware


def _make_request(name: str, args: dict, call_id: str = "call-1") -> SimpleNamespace:
    """Build a minimal ToolCallRequest-shaped object for the middleware."""
    return SimpleNamespace(
        tool_call={"name": name, "args": args, "id": call_id}
    )


def test_tool_args_key_is_stable_across_arg_order():
    middleware = ToolRetryMiddleware()
    a = _make_request("get_support_article_content", {"x": 1, "y": 2})
    b = _make_request("get_support_article_content", {"y": 2, "x": 1})

    assert middleware._tool_args_key(a) == middleware._tool_args_key(b)


def test_short_circuits_after_threshold_of_identical_calls():
    middleware = ToolRetryMiddleware(max_identical_calls=2)
    call_count = {"n": 0}

    async def handler(request):
        call_count["n"] += 1
        return "ERROR_ARTICLE_NOT_FOUND: nope"

    async def run():
        results = []
        for i in range(4):
            req = _make_request(
                "get_support_article_content", {"article_id": "a1"}, call_id=f"c{i}"
            )
            results.append(await middleware.awrap_tool_call(req, handler))
        return results

    results = asyncio.run(run())

    # First two identical calls invoke the handler; the third and fourth are
    # short-circuited with a terminal message.
    assert call_count["n"] == 2
    terminal = results[-1]
    assert "already attempted" in terminal.content
    assert terminal.name == "get_support_article_content"


def test_agent_level_cap_halts_repeated_identical_loop_for_any_tool():
    # The middleware wraps every tool the agent has, so max_identical_calls is
    # the global cap: even a non-pylon tool cannot loop past the threshold.
    middleware = ToolRetryMiddleware(max_identical_calls=2)
    handler_calls = {"n": 0}

    async def handler(request):
        handler_calls["n"] += 1
        return "ok"

    async def run():
        last = None
        for i in range(50):
            req = _make_request("check_links", {"urls": ["u1"]}, call_id=f"c{i}")
            last = await middleware.awrap_tool_call(req, handler)
        return last

    last = asyncio.run(run())

    assert handler_calls["n"] == 2
    assert "already attempted" in last.content


def test_distinct_args_are_not_short_circuited():
    middleware = ToolRetryMiddleware(max_identical_calls=2)

    async def handler(request):
        return "ok"

    async def run():
        out = []
        for aid in ("a1", "a2", "a3", "a4"):
            req = _make_request("get_support_article_content", {"article_id": aid})
            out.append(await middleware.awrap_tool_call(req, handler))
        return out

    results = asyncio.run(run())
    assert all(r == "ok" for r in results)
