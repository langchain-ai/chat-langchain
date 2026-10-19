"""Tests for guardrails decision metadata on child and root runs."""

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

os.environ["USE_LOCAL_PROMPTS"] = "1"

from src.middleware import guardrails_middleware as guardrails_module
from src.middleware.guardrails_middleware import GuardrailsMiddleware


def _run_tree(parent_run=None, trace_id=None):
    run_id = uuid4()
    return SimpleNamespace(
        id=run_id,
        trace_id=trace_id or (parent_run.trace_id if parent_run else run_id),
        parent_run=parent_run,
        metadata={"existing": "preserved"},
    )


def _middleware():
    return GuardrailsMiddleware.__new__(GuardrailsMiddleware)


@pytest.mark.parametrize("result", ["ALLOWED", "BLOCKED"])
def test_decision_metadata_reaches_child_and_root(monkeypatch, result):
    root = _run_tree()
    parent = _run_tree(root)
    child = _run_tree(parent)
    monkeypatch.setattr(guardrails_module.ls, "get_current_run_tree", lambda: child)
    client = Mock()
    monkeypatch.setattr(guardrails_module, "Client", client)

    _middleware()._track_decision_metadata(
        {"decision": result, "explanation": "Classifier explanation."}
    )

    expected = {
        "existing": "preserved",
        "guardrails_result": result,
        "guardrails_explanation": "Classifier explanation.",
    }
    assert child.metadata == root.metadata == expected
    assert parent.metadata == {"existing": "preserved"}
    client.assert_not_called()


def test_emitted_block_rule_is_bounded_on_root(monkeypatch):
    root = _run_tree()
    child = _run_tree(root)
    monkeypatch.setattr(guardrails_module.ls, "get_current_run_tree", lambda: child)
    rule = "Creative writing" * 30

    _middleware()._track_decision_metadata(
        {"decision": "BLOCKED", "explanation": "Off topic.", "rule": rule}
    )

    assert root.metadata["guardrails_rule"] == rule[:200]
    assert "guardrails_rule" not in child.metadata


@pytest.mark.parametrize("rule", [None, "", "   ", 123])
def test_invalid_rule_is_not_recorded(monkeypatch, rule):
    root = _run_tree()
    monkeypatch.setattr(guardrails_module.ls, "get_current_run_tree", lambda: root)

    _middleware()._track_decision_metadata(
        {"decision": "BLOCKED", "explanation": "Off topic.", "rule": rule}
    )

    assert "guardrails_rule" not in root.metadata


def test_current_root_keeps_decision_metadata_without_remote_update(monkeypatch):
    root = _run_tree()
    monkeypatch.setattr(guardrails_module.ls, "get_current_run_tree", lambda: root)
    client = Mock()
    monkeypatch.setattr(guardrails_module, "Client", client)

    _middleware()._track_decision_metadata(
        {"decision": "ALLOWED", "explanation": "In scope.", "rule": "Not a block rule"}
    )

    assert root.metadata == {
        "existing": "preserved",
        "guardrails_result": "ALLOWED",
        "guardrails_explanation": "In scope.",
    }
    client.assert_not_called()


@pytest.mark.parametrize("links", ["missing", "detached", "partial"])
def test_unavailable_parent_links_update_trace_root(monkeypatch, links):
    trace_id = uuid4()
    parent = _run_tree(trace_id=trace_id) if links == "partial" else None
    child = _run_tree(parent, trace_id=trace_id)
    if links == "missing":
        del child.parent_run
    monkeypatch.setattr(guardrails_module.ls, "get_current_run_tree", lambda: child)
    client = Mock()
    monkeypatch.setattr(guardrails_module, "Client", lambda: client)

    _middleware()._track_decision_metadata(
        {"decision": "BLOCKED", "explanation": "Off topic.", "rule": "Fiction"}
    )

    assert child.metadata == {
        "existing": "preserved",
        "guardrails_result": "BLOCKED",
        "guardrails_explanation": "Off topic.",
    }
    client.update_run.assert_called_once_with(
        trace_id,
        extra={
            "metadata": {
                "guardrails_result": "BLOCKED",
                "guardrails_explanation": "Off topic.",
                "guardrails_rule": "Fiction",
            }
        },
    )


@pytest.mark.parametrize("fallback_fails", [False, True])
def test_blocked_turn_survives_tracing_update_failure(monkeypatch, fallback_fails):
    root = _run_tree()
    child = _run_tree(root)
    if fallback_fails:
        child.parent_run = None
    monkeypatch.setattr(guardrails_module.ls, "get_current_run_tree", lambda: child)
    client = Mock()
    client.update_run.side_effect = RuntimeError("Tracing unavailable")
    monkeypatch.setattr(guardrails_module, "Client", lambda: client)
    middleware = _middleware()
    middleware.block_off_topic = True
    middleware._classify_query = AsyncMock(
        return_value={
            "decision": "BLOCKED",
            "explanation": "Off topic.",
            "rule": "Creative writing",
        }
    )
    middleware._generate_rejection_message = AsyncMock(
        return_value=AIMessage(content="Please ask about LangChain.")
    )
    middleware._add_to_dataset = AsyncMock()

    result = asyncio.run(
        middleware.abefore_agent(
            {"messages": [HumanMessage(content="Write a poem.")]},
            Runtime(context=None),
        )
    )

    assert result["jump_to"] == "end"
    assert result["off_topic_query"] is True
    assert result["messages"][0].content == "Please ask about LangChain."
    assert child.metadata["guardrails_result"] == "BLOCKED"
    assert child.metadata["guardrails_explanation"] == "Off topic."
    if fallback_fails:
        client.update_run.assert_called_once()
    else:
        client.update_run.assert_not_called()
        assert root.metadata["guardrails_result"] == "BLOCKED"
        assert root.metadata["guardrails_explanation"] == "Off topic."
        assert root.metadata["guardrails_rule"] == "Creative writing"


@pytest.mark.parametrize("lookup_fails", [False, True])
def test_unavailable_tracing_is_nonfatal(monkeypatch, lookup_fails):
    lookup = Mock(return_value=None)
    if lookup_fails:
        lookup.side_effect = RuntimeError("Tracing unavailable")
    monkeypatch.setattr(guardrails_module.ls, "get_current_run_tree", lookup)
    client = Mock()
    monkeypatch.setattr(guardrails_module, "Client", client)

    _middleware()._track_decision_metadata(
        {"decision": "ALLOWED", "explanation": "In scope."}
    )

    client.assert_not_called()
