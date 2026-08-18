"""Tests for the hard context-window guard around the model call."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.middleware.context_overflow_middleware import (
    GRACEFUL_OVERFLOW_MESSAGE,
    ContextOverflowGuardMiddleware,
)


class _OverflowError(Exception):
    """Stand-in whose class name matches the OpenAI overflow error."""


_OverflowError.__name__ = "OpenAIContextOverflowError"


def _request(messages):
    def override(*, messages):
        return _request(messages)

    return SimpleNamespace(
        model=SimpleNamespace(profile={"max_input_tokens": 1000}),
        system_prompt="sys",
        messages=messages,
        override=override,
    )


def _big_tool_message() -> ToolMessage:
    return ToolMessage(content="X" * 20_000, tool_call_id="t1", name="grep")


def test_trims_prompt_to_budget_before_model_call():
    middleware = ContextOverflowGuardMiddleware(
        default_max_input_tokens=1000,
        input_budget_fraction=0.5,
        max_tool_result_chars=200,
    )
    request = _request([HumanMessage(content="hi"), _big_tool_message()])
    seen: dict = {}

    async def handler(req):
        seen["tokens"] = middleware._count(req)
        return AIMessage(content="ok")

    result = asyncio.run(middleware.awrap_model_call(request, handler))

    assert result.content == "ok"
    assert seen["tokens"] <= middleware._budget(request)


def test_recovers_from_overflow_with_trim_and_retry():
    middleware = ContextOverflowGuardMiddleware(default_max_input_tokens=1000)
    request = _request([HumanMessage(content="hi"), _big_tool_message()])
    calls = {"n": 0}

    async def handler(req):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _OverflowError("Error code: 400 context_length_exceeded")
        return AIMessage(content="recovered")

    result = asyncio.run(middleware.awrap_model_call(request, handler))

    assert result.content == "recovered"
    assert calls["n"] == 2


def test_terminates_gracefully_when_overflow_persists():
    middleware = ContextOverflowGuardMiddleware(default_max_input_tokens=1000)
    request = _request([HumanMessage(content="hi"), _big_tool_message()])

    async def handler(req):
        raise _OverflowError("Error code: 400 context_length_exceeded")

    result = asyncio.run(middleware.awrap_model_call(request, handler))

    assert isinstance(result, AIMessage)
    assert result.content == GRACEFUL_OVERFLOW_MESSAGE


def test_non_overflow_errors_propagate():
    middleware = ContextOverflowGuardMiddleware(default_max_input_tokens=1000)
    request = _request([HumanMessage(content="hi")])

    async def handler(req):
        raise ValueError("boom")

    try:
        asyncio.run(middleware.awrap_model_call(request, handler))
    except ValueError:
        return
    raise AssertionError("expected ValueError to propagate")
