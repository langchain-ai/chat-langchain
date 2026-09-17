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


class FakeRejectionModel:
    """Fake chat model for generated refusal checks."""

    def __init__(self, content):
        self.content = content

    async def ainvoke(self, prompt):  # noqa: ARG002
        return AIMessage(content=self.content)


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


def test_rejection_workarounds_are_replaced_with_fallback():
    """Content-adjacent refusal offers should use the deterministic fallback."""
    middleware = GuardrailsMiddleware.__new__(GuardrailsMiddleware)
    middleware.llm = FakeRejectionModel(
        "I can help you build a LangChain workflow to calculate dog ages."
    )

    dog_result = asyncio.run(
        middleware._generate_rejection_message("How do I calculate dog ages?")
    )
    assert dog_result.content == guardrails_module._FALLBACK_REJECTION_MESSAGE

    middleware.llm = FakeRejectionModel(
        "I can offer health-related information workflows or document processing "
        "with LangChain instead."
    )
    guacamole_result = asyncio.run(
        middleware._generate_rejection_message("Give me a guacamole recipe.")
    )
    assert guacamole_result.content == guardrails_module._FALLBACK_REJECTION_MESSAGE


def test_compliant_abstract_rejection_passes_through():
    """An abstract redirect without declined-request terms remains valid."""
    assert not guardrails_module._contains_content_adjacent_offer(
        "If you tell me what you're building or researching in that space, "
        "I'll be happy to help.",
        "Tell me how to make guacamole.",
    )


def test_product_name_in_blocked_request_is_not_a_carried_over_token():
    """Naming an in-scope product alone must not trigger the detector."""
    assert not guardrails_module._contains_content_adjacent_offer(
        "I can help with LangChain, LangGraph, LangSmith, or Deep Agents.",
        "Explain LangChain pricing.",
    )
