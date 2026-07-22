"""Tests for guardrails model fallback behavior."""

import asyncio
import json
import os

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

os.environ["USE_LOCAL_PROMPTS"] = "1"

from src.middleware import guardrails_middleware as guardrails_module
from src.middleware.guardrails_middleware import (
    GuardrailsClassificationError,
    GuardrailsMiddleware,
    _is_guardrails_decision_message,
)


class FakeStructuredModel:
    """Fake structured model that returns or raises queued outcomes."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def with_structured_output(self, schema):  # noqa: ARG002
        return self

    async def ainvoke(self, prompt, config=None):  # noqa: ARG002
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _middleware_with_models(*models: tuple[str, FakeStructuredModel]) -> GuardrailsMiddleware:
    middleware = GuardrailsMiddleware.__new__(GuardrailsMiddleware)
    middleware.classifier_llms = list(models)
    middleware.block_off_topic = True
    return middleware


def test_guardrails_falls_back_after_primary_retries(monkeypatch):
    """The fallback model should get its own retry budget after primary fails."""
    monkeypatch.setattr(guardrails_module, "GUARDRAILS_MAX_RETRIES", 1)

    primary = FakeStructuredModel([RuntimeError("primary down"), RuntimeError("still down")])
    fallback = FakeStructuredModel(
        [{"decision": "ALLOWED", "explanation": "LangChain-related question."}]
    )
    middleware = _middleware_with_models(("primary", primary), ("fallback", fallback))

    result = asyncio.run(
        middleware._classify_query([HumanMessage(content="How do agents work?")])
    )

    assert result["decision"] == "ALLOWED"
    assert primary.calls == 2
    assert fallback.calls == 1


def test_guardrails_raises_after_all_models_exhaust_retries(monkeypatch):
    """Guardrails should fail only after every model exhausts retries."""
    monkeypatch.setattr(guardrails_module, "GUARDRAILS_MAX_RETRIES", 1)

    primary = FakeStructuredModel([RuntimeError("primary down"), RuntimeError("still down")])
    fallback = FakeStructuredModel(
        [RuntimeError("fallback down"), RuntimeError("fallback still down")]
    )
    middleware = _middleware_with_models(("primary", primary), ("fallback", fallback))

    with pytest.raises(GuardrailsClassificationError):
        asyncio.run(
            middleware._classify_query([HumanMessage(content="How do agents work?")])
        )

    assert primary.calls == 2
    assert fallback.calls == 2


def test_guardrails_all_failed_classification_allows_main_agent(monkeypatch):
    """If guardrails classification fully fails, the main agent should continue."""
    middleware = _middleware_with_models()

    async def _raise_classification_error(messages):  # noqa: ARG001
        raise GuardrailsClassificationError("all models failed")

    monkeypatch.setattr(middleware, "_classify_query", _raise_classification_error)

    result = asyncio.run(
        middleware.abefore_agent(
            {"messages": [HumanMessage(content="How do agents work?")]},
            Runtime(context=None),
        )
    )

    assert result == {"off_topic_query": False}


def test_is_guardrails_decision_message_detects_classifier_json():
    """The helper should only flag AIMessages that are serialized GuardrailsDecisions."""
    decision_json = json.dumps(
        {"decision": "ALLOWED", "explanation": "LangChain question."}
    )
    assert _is_guardrails_decision_message(AIMessage(content=decision_json)) is True
    assert _is_guardrails_decision_message(AIMessage(content="Here is your answer.")) is False
    assert _is_guardrails_decision_message(HumanMessage(content=decision_json)) is False


def test_allowed_query_strips_contaminating_decision_message(monkeypatch):
    """An ALLOWED in-scope query must not surface the raw guardrail decision JSON."""
    middleware = _middleware_with_models()

    async def _allow(messages):  # noqa: ARG001
        return {"decision": "ALLOWED", "explanation": "LangChain-related question."}

    monkeypatch.setattr(middleware, "_classify_query", _allow)

    decision_message = AIMessage(
        content=json.dumps(
            {"decision": "ALLOWED", "explanation": "LangChain-related question."}
        )
    )
    result = asyncio.run(
        middleware.abefore_agent(
            {
                "messages": [
                    HumanMessage(content="How do agents work?"),
                    decision_message,
                ]
            },
            Runtime(context=None),
        )
    )

    assert result is not None
    remaining = result["messages"]
    assert decision_message not in remaining
    assert not any(_is_guardrails_decision_message(m) for m in remaining)


def test_allowed_query_without_contamination_returns_none(monkeypatch):
    """A clean ALLOWED query should proceed to the agent without altering messages."""
    middleware = _middleware_with_models()

    async def _allow(messages):  # noqa: ARG001
        return {"decision": "ALLOWED", "explanation": "LangChain-related question."}

    monkeypatch.setattr(middleware, "_classify_query", _allow)

    result = asyncio.run(
        middleware.abefore_agent(
            {"messages": [HumanMessage(content="How do agents work?")]},
            Runtime(context=None),
        )
    )

    assert result is None
