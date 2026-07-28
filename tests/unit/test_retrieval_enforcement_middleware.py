"""Tests for the runtime enforcement of the docs agent's retrieval mandate."""

from __future__ import annotations

import os
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

os.environ["USE_LOCAL_PROMPTS"] = "1"

from src.middleware.retrieval_enforcement_middleware import (
    FORCE_RETRIEVAL_INSTRUCTION,
    RetrievalEnforcementMiddleware,
)

RUNTIME = SimpleNamespace()


def _search_call(message_id: str = "a-search") -> AIMessage:
    return AIMessage(
        content="",
        id=message_id,
        tool_calls=[
            {
                "name": "search_docs_by_lang_chain",
                "args": {"query": "middleware"},
                "id": "call-1",
            }
        ],
    )


def _search_result(message_id: str = "t-search") -> ToolMessage:
    return ToolMessage(
        content="Middleware guide",
        id=message_id,
        name="search_docs_by_lang_chain",
        tool_call_id="call-1",
    )


def test_code_block_answer_without_retrieval_is_blocked():
    middleware = RetrievalEnforcementMiddleware()
    state = {
        "messages": [
            HumanMessage(content="How do I add middleware?", id="h1"),
            AIMessage(content="```python\nagent = create_agent()\n```", id="a1"),
        ]
    }

    update = middleware.after_model(state, runtime=RUNTIME)

    assert update is not None
    assert update["jump_to"] == "model"
    assert update["messages"][0].content == FORCE_RETRIEVAL_INSTRUCTION


def test_symbol_answer_without_retrieval_is_blocked():
    middleware = RetrievalEnforcementMiddleware()
    state = {
        "messages": [
            HumanMessage(content="Does this work?", id="h1"),
            AIMessage(
                content="Pass create_deep_agent(subagents=[...]) and you are set.",
                id="a1",
            ),
        ]
    }

    update = middleware.after_model(state, runtime=RUNTIME)

    assert update is not None
    assert update["jump_to"] == "model"


def test_answer_is_allowed_when_retrieval_ran_this_turn():
    middleware = RetrievalEnforcementMiddleware()
    state = {
        "messages": [
            HumanMessage(content="How do I add middleware?", id="h1"),
            _search_call(),
            _search_result(),
            AIMessage(content="```python\nagent = create_agent()\n```", id="a1"),
        ]
    }

    assert middleware.after_model(state, runtime=RUNTIME) is None


def test_retrieval_on_earlier_turn_does_not_ground_this_turn():
    middleware = RetrievalEnforcementMiddleware()
    state = {
        "messages": [
            HumanMessage(content="How do I add middleware?", id="h1"),
            _search_call(),
            _search_result(),
            AIMessage(content="Here is the middleware guide.", id="a1"),
            HumanMessage(content="Now show me the subagent schema.", id="h2"),
            AIMessage(content="```python\ncreate_deep_agent(subagents=[])\n```", id="a2"),
        ]
    }

    update = middleware.after_model(state, runtime=RUNTIME)

    assert update is not None
    assert update["jump_to"] == "model"


def test_greeting_answer_is_allowed_without_tools():
    middleware = RetrievalEnforcementMiddleware()
    state = {
        "messages": [
            HumanMessage(content="hi there", id="h1"),
            AIMessage(content="Hello! What can I help you with today?", id="a1"),
        ]
    }

    assert middleware.after_model(state, runtime=RUNTIME) is None


def test_forced_retry_is_capped_at_one_round():
    middleware = RetrievalEnforcementMiddleware()
    draft = AIMessage(content="```python\ncreate_agent()\n```", id="a2")
    state = {
        "messages": [
            HumanMessage(content="How do I add middleware?", id="h1"),
            AIMessage(content="```python\ncreate_agent()\n```", id="a1"),
            HumanMessage(content=FORCE_RETRIEVAL_INSTRUCTION, id="h2"),
            draft,
        ]
    }

    assert middleware.after_model(state, runtime=RUNTIME) is None


def test_draft_with_tool_calls_is_ignored():
    middleware = RetrievalEnforcementMiddleware()
    state = {
        "messages": [
            HumanMessage(content="How do I add middleware?", id="h1"),
            _search_call("a1"),
        ]
    }

    assert middleware.after_model(state, runtime=RUNTIME) is None
