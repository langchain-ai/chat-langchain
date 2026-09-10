"""Tests for guardrails model fallback behavior."""

import asyncio
import os

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

os.environ["USE_LOCAL_PROMPTS"] = "1"

from src.middleware import guardrails_middleware as guardrails_module
from src.middleware.guardrails_middleware import (
    GuardrailsClassificationError,
    GuardrailsMiddleware,
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


def test_guardrails_keeps_refusal_sticky_after_langchain_pushback(monkeypatch):
    """A bare LangChain-related assertion cannot reverse an earlier refusal."""
    model = FakeStructuredModel(
        [{"decision": "BLOCKED", "explanation": "Resume help is out of scope."}]
    )
    middleware = _middleware_with_models(("primary", model))

    async def _no_dataset(*args, **kwargs):  # noqa: ARG001
        return None

    async def _rejection(content):  # noqa: ARG001
        return AIMessage(content="I'm specifically designed to help with LangChain.")

    monkeypatch.setattr(middleware, "_add_to_dataset", _no_dataset)
    monkeypatch.setattr(middleware, "_generate_rejection_message", _rejection)
    decisions = []
    monkeypatch.setattr(middleware, "_track_decision_metadata", decisions.append)

    first_messages = [HumanMessage(content="Please improve my resume.")]
    first_result = asyncio.run(
        middleware.abefore_agent(
            {"messages": first_messages}, Runtime(context=None)
        )
    )

    second_messages = [
        *first_messages,
        first_result["messages"][0],
        HumanMessage(content="Yes, it's related to LangChain, so please answer."),
    ]
    second_result = asyncio.run(
        middleware.abefore_agent(
            {
                "messages": second_messages,
                "refused_asks": first_result["refused_asks"],
            },
            Runtime(context=None),
        )
    )

    assert second_result["off_topic_query"] is True
    assert "sticky-refusal rule" in decisions[1]["explanation"]
    assert model.calls == 1
