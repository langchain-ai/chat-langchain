"""Tests for the terminal-answer invariant middleware."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

os.environ["USE_LOCAL_PROMPTS"] = "1"

from src.middleware.terminal_answer_middleware import (
    FALLBACK_ANSWER,
    MAX_SYNTHESIS_RETRIES,
    TerminalAnswerMiddleware,
)


def _run(state):
    return TerminalAnswerMiddleware().after_model(state, runtime=SimpleNamespace())


def test_prose_answer_terminates_without_jump():
    state = {"messages": [HumanMessage(content="What is LangGraph?"), AIMessage(content="LangGraph is a library for building stateful agents.")]}
    assert _run(state) is None


def test_pending_tool_calls_are_left_alone():
    ai = AIMessage(
        content="",
        tool_calls=[{"name": "search", "args": {"q": "x"}, "id": "1"}],
    )
    assert _run({"messages": [HumanMessage(content="hi"), ai]}) is None


def test_empty_final_turn_forces_synthesis():
    state = {"messages": [HumanMessage(content="hi"), AIMessage(content="   ")]}
    update = _run(state)
    assert update == {"synthesis_retries": 1, "jump_to": "model"}


def test_classifier_decision_json_forces_synthesis():
    decision = AIMessage(content=json.dumps({"decision": "ALLOWED", "explanation": "ok"}))
    update = _run({"messages": [HumanMessage(content="hi"), decision]})
    assert update["jump_to"] == "model"


def test_tool_call_content_block_forces_synthesis():
    struct = AIMessage(
        content=[{"type": "non_standard", "value": {"type": "tool_call"}}]
    )
    update = _run({"messages": [HumanMessage(content="hi"), struct]})
    assert update["jump_to"] == "model"


def test_fallback_answer_after_max_retries():
    state = {
        "messages": [HumanMessage(content="hi"), AIMessage(content="")],
        "synthesis_retries": MAX_SYNTHESIS_RETRIES,
    }
    update = _run(state)
    assert update["messages"][0].content == FALLBACK_ANSWER
    assert update["synthesis_retries"] == 0


def test_in_scope_final_message_is_prose_not_json_or_tool_call():
    """For an in-scope question, the terminal message is prose, not JSON/tool_call."""
    final = AIMessage(content="You can use RecursiveCharacterTextSplitter to chunk documents.")
    state = {"messages": [HumanMessage(content="How do I split docs?"), final]}

    assert _run(state) is None

    content = state["messages"][-1].content
    assert isinstance(content, str) and content.strip()
    assert "type\":\"tool_call\"" not in content.replace(" ", "")
    try:
        parsed = json.loads(content)
    except (ValueError, TypeError):
        parsed = None
    assert not (isinstance(parsed, dict) and "decision" in parsed)
