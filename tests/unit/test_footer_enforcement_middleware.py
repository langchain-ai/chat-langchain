"""Tests for the output-stage footer enforcement middleware."""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

os.environ["USE_LOCAL_PROMPTS"] = "1"

from src.middleware.footer_enforcement_middleware import (
    MIN_SUBSTANTIVE_CHARS,
    FooterEnforcementMiddleware,
)


def _middleware() -> FooterEnforcementMiddleware:
    return FooterEnforcementMiddleware.__new__(FooterEnforcementMiddleware)


def test_short_answer_is_untouched():
    mw = _middleware()
    state = {"messages": [AIMessage(content="Thanks, glad it helped!")]}
    assert asyncio.run(mw.aafter_agent(state, runtime=SimpleNamespace())) is None


def test_answer_with_footer_is_untouched():
    mw = _middleware()
    body = "x" * (MIN_SUBSTANTIVE_CHARS + 1) + "\n\n**Relevant docs:**\n\n- a"
    state = {"messages": [AIMessage(content=body)]}
    assert asyncio.run(mw.aafter_agent(state, runtime=SimpleNamespace())) is None


def test_no_tool_urls_passes_through():
    mw = _middleware()
    body = "x" * (MIN_SUBSTANTIVE_CHARS + 1)
    state = {"messages": [HumanMessage(content="hi"), AIMessage(content=body)]}
    assert asyncio.run(mw.aafter_agent(state, runtime=SimpleNamespace())) is None


def test_missing_footer_triggers_rewrite():
    mw = _middleware()
    body = "x" * (MIN_SUBSTANTIVE_CHARS + 1)
    tool_msg = ToolMessage(
        content="https://docs.langchain.com/a https://docs.langchain.com/b",
        tool_call_id="t1",
    )
    final = AIMessage(content=body, id="a1")

    async def _fake_append(text, urls):
        return text + "\n\n**Relevant docs:**\n\n- [A](https://docs.langchain.com/a)"

    mw._append_footer = _fake_append

    update = asyncio.run(
        mw.aafter_agent({"messages": [tool_msg, final]}, runtime=SimpleNamespace())
    )

    assert update is not None
    assert "**Relevant docs:**" in update["messages"][0].content
    assert update["messages"][0].id == "a1"
