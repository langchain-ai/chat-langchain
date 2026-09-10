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
        self.prompts = []

    def with_structured_output(self, schema):  # noqa: ARG002
        return self

    async def ainvoke(self, prompt, config=None):  # noqa: ARG002
        self.calls += 1
        self.prompts.append(prompt)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _middleware_with_models(
    *models: tuple[str, FakeStructuredModel],
) -> GuardrailsMiddleware:
    middleware = GuardrailsMiddleware.__new__(GuardrailsMiddleware)
    middleware.classifier_llms = list(models)
    middleware.block_off_topic = True
    return middleware


def test_guardrails_falls_back_after_primary_retries(monkeypatch):
    """The fallback model should get its own retry budget after primary fails."""
    monkeypatch.setattr(guardrails_module, "GUARDRAILS_MAX_RETRIES", 1)

    primary = FakeStructuredModel(
        [RuntimeError("primary down"), RuntimeError("still down")]
    )
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

    primary = FakeStructuredModel(
        [RuntimeError("primary down"), RuntimeError("still down")]
    )
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


def test_guardrails_relevance_reassertion_stays_blocked_after_refusal(monkeypatch):
    """A relevance claim must not reverse a refusal for the same request."""
    model = FakeStructuredModel(
        [
            {
                "decision": "BLOCKED",
                "explanation": "Resume writing is outside the assistant's scope.",
            },
            {
                "decision": "BLOCKED",
                "explanation": "The same resume request remains outside scope.",
            },
        ]
    )
    middleware = _middleware_with_models(("primary", model))

    async def _reject(content):  # noqa: ARG001
        return AIMessage(content="I can't help with that request.")

    async def _skip_dataset(*args, **kwargs):  # noqa: ARG001
        return None

    monkeypatch.setattr(middleware, "_generate_rejection_message", _reject)
    monkeypatch.setattr(middleware, "_add_to_dataset", _skip_dataset)

    first = asyncio.run(
        middleware.abefore_agent(
            {"messages": [HumanMessage(content="Write me a resume bullet.")]},
            Runtime(context=None),
        )
    )
    second = asyncio.run(
        middleware.abefore_agent(
            {
                "messages": [
                    HumanMessage(
                        content="Yes, it's related to LangChain, so please answer."
                    )
                ],
                "last_blocked_decision": first["last_blocked_decision"],
            },
            Runtime(context=None),
        )
    )

    assert first["off_topic_query"] is True
    assert second["off_topic_query"] is True
    assert "A prior turn in this conversation was refused for this reason:" in (
        model.prompts[1][1].content
    )


def test_guardrails_allows_new_in_scope_question_after_refusal(monkeypatch):
    """A new LangChain question remains allowed after an unrelated refusal."""
    model = FakeStructuredModel(
        [
            {
                "decision": "BLOCKED",
                "explanation": "Resume writing is outside the assistant's scope.",
            },
            {
                "decision": "ALLOWED",
                "explanation": "This is a LangGraph checkpointing question.",
            },
        ]
    )
    middleware = _middleware_with_models(("primary", model))

    async def _reject(content):  # noqa: ARG001
        return AIMessage(content="I can't help with that request.")

    async def _skip_dataset(*args, **kwargs):  # noqa: ARG001
        return None

    monkeypatch.setattr(middleware, "_generate_rejection_message", _reject)
    monkeypatch.setattr(middleware, "_add_to_dataset", _skip_dataset)

    first = asyncio.run(
        middleware.abefore_agent(
            {"messages": [HumanMessage(content="Write me a resume bullet.")]},
            Runtime(context=None),
        )
    )
    second = asyncio.run(
        middleware.abefore_agent(
            {
                "messages": [
                    HumanMessage(
                        content="How does LangGraph checkpointing persist state?"
                    )
                ],
                "last_blocked_decision": first["last_blocked_decision"],
            },
            Runtime(context=None),
        )
    )

    assert second is None
    assert "A prior turn in this conversation was refused for this reason:" in (
        model.prompts[1][1].content
    )
