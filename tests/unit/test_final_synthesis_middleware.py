"""Tests for the final-answer synthesis safety-net middleware."""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

os.environ["USE_LOCAL_PROMPTS"] = "1"

from src.middleware.final_synthesis_middleware import (
    FinalSynthesisMiddleware,
    needs_final_answer,
)


class _StubModel:
    def __init__(self, response):
        self._response = response
        self.calls = 0

    async def ainvoke(self, messages, *args, **kwargs):
        self.calls += 1
        return self._response


def _run(coro):
    return asyncio.run(coro)


def test_needs_final_answer_on_tool_result():
    messages = [
        HumanMessage(content="How do I use LangGraph?"),
        AIMessage(content="", tool_calls=[{"name": "search", "args": {}, "id": "1"}]),
        ToolMessage(content="docs...", tool_call_id="1"),
    ]
    assert needs_final_answer(messages) is True


def test_needs_final_answer_on_empty_ai_with_tool_calls():
    messages = [
        HumanMessage(content="q"),
        AIMessage(content="", tool_calls=[{"name": "search", "args": {}, "id": "1"}]),
    ]
    assert needs_final_answer(messages) is True


def test_no_final_answer_needed_when_ends_with_text_ai():
    messages = [
        HumanMessage(content="q"),
        ToolMessage(content="docs...", tool_call_id="1"),
        AIMessage(content="Here is your grounded answer."),
    ]
    assert needs_final_answer(messages) is False


def test_after_agent_synthesizes_final_answer_from_tool_result():
    middleware = FinalSynthesisMiddleware(
        model=_StubModel(AIMessage(content="Grounded final answer."))
    )
    state = {
        "messages": [
            HumanMessage(content="How do I use LangGraph?"),
            AIMessage(
                content="", tool_calls=[{"name": "search", "args": {}, "id": "1"}]
            ),
            ToolMessage(content="relevant docs", tool_call_id="1"),
        ]
    }

    update = _run(middleware.aafter_agent(state, runtime=SimpleNamespace()))

    assert update is not None
    final = update["messages"][-1]
    assert final.type == "ai"
    assert final.content == "Grounded final answer."
    assert not getattr(final, "tool_calls", None)


def test_after_agent_noop_when_answer_already_present():
    stub = _StubModel(AIMessage(content="should not be called"))
    middleware = FinalSynthesisMiddleware(model=stub)
    state = {
        "messages": [
            HumanMessage(content="q"),
            AIMessage(content="Already answered."),
        ]
    }

    assert _run(middleware.aafter_agent(state, runtime=SimpleNamespace())) is None
    assert stub.calls == 0
