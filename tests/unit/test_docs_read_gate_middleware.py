"""Tests for the read-before-answer gate on the docs agent."""

from __future__ import annotations

import os
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

os.environ["USE_LOCAL_PROMPTS"] = "1"

from src.middleware.docs_read_gate_middleware import (
    READ_NUDGE,
    SUBSTANTIVE_CHARS,
    DocsReadGateMiddleware,
)

RUNTIME = SimpleNamespace()


def _search_call(name="search_docs_by_lang_chain", call_id="c1"):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": {"query": "middleware"}, "id": call_id}],
    )


def test_nudges_when_answer_rests_on_search_titles_only():
    middleware = DocsReadGateMiddleware()
    state = {
        "messages": [
            HumanMessage(content="How do I add middleware?"),
            _search_call(),
            ToolMessage(content="Page: /oss/python/middleware", tool_call_id="c1"),
            AIMessage(content="**Use middleware.**\n\n```python\nfoo()\n```"),
        ]
    }

    update = middleware.after_model(state, RUNTIME)

    assert update is not None
    assert update["jump_to"] == "model"
    assert update["messages"][0].content == READ_NUDGE


def test_no_nudge_when_page_content_was_read():
    middleware = DocsReadGateMiddleware()
    state = {
        "messages": [
            HumanMessage(content="How do I add middleware?"),
            _search_call(),
            ToolMessage(content="Page: /oss/python/middleware", tool_call_id="c1"),
            _search_call(name="query_docs_filesystem_docs_by_lang_chain", call_id="c2"),
            ToolMessage(content="class AgentMiddleware: ...", tool_call_id="c2"),
            AIMessage(content="**Use middleware.**\n\n```python\nfoo()\n```"),
        ]
    }

    assert middleware.after_model(state, RUNTIME) is None


def test_no_nudge_when_grep_read_page_content():
    middleware = DocsReadGateMiddleware()
    state = {
        "messages": [
            HumanMessage(content="How do I add middleware?"),
            _search_call(),
            ToolMessage(content="Page: /oss/python/middleware", tool_call_id="c1"),
            _search_call(name="grep", call_id="c2"),
            ToolMessage(content="class AgentMiddleware: ...", tool_call_id="c2"),
            AIMessage(content="x" * (SUBSTANTIVE_CHARS + 1)),
        ]
    }

    assert middleware.after_model(state, RUNTIME) is None


def test_no_nudge_for_short_answer():
    middleware = DocsReadGateMiddleware()
    state = {
        "messages": [
            HumanMessage(content="hey"),
            _search_call(),
            ToolMessage(content="Page: /oss/python/middleware", tool_call_id="c1"),
            AIMessage(content="**Hi!** What can I help you with?"),
        ]
    }

    assert middleware.after_model(state, RUNTIME) is None


def test_no_nudge_while_model_is_still_calling_tools():
    middleware = DocsReadGateMiddleware()
    state = {"messages": [HumanMessage(content="middleware?"), _search_call()]}

    assert middleware.after_model(state, RUNTIME) is None


def test_nudge_injected_only_once_per_turn():
    middleware = DocsReadGateMiddleware()
    state = {
        "messages": [
            HumanMessage(content="How do I add middleware?"),
            _search_call(),
            ToolMessage(content="Page: /oss/python/middleware", tool_call_id="c1"),
            AIMessage(content="```python\nfoo()\n```"),
            SystemMessage(content=READ_NUDGE),
            AIMessage(content="```python\nfoo()\n```"),
        ]
    }

    assert middleware.after_model(state, RUNTIME) is None


def test_handles_content_blocks_and_empty_state():
    middleware = DocsReadGateMiddleware()
    blocks = {
        "messages": [
            HumanMessage(content="How do I add middleware?"),
            _search_call(),
            ToolMessage(content="Page: /oss/python/middleware", tool_call_id="c1"),
            AIMessage(content=[{"type": "text", "text": "```python\nfoo()\n```"}]),
        ]
    }

    assert middleware.after_model(blocks, RUNTIME) is not None
    assert middleware.after_model({"messages": []}, RUNTIME) is None
    assert middleware.after_model({}, RUNTIME) is None
