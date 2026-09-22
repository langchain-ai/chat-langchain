"""Tests for guardrails model fallback behavior."""

import asyncio
import os

import pytest
from langchain_core.messages import HumanMessage
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
    """Fake model that captures the rejection prompt."""

    def __init__(self):
        self.prompt = None

    async def ainvoke(self, prompt):
        self.prompt = prompt
        return HumanMessage(content="A rejection response.")


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


def test_blocked_rejection_prompt_includes_classifier_explanation(monkeypatch):
    """The blocked query's policy reason should reach the rejection model."""
    middleware = _middleware_with_models()
    rejection_model = FakeRejectionModel()
    middleware.llm = rejection_model

    async def _blocked_query(messages, guardrail_history=None):  # noqa: ARG001
        return {
            "decision": "BLOCKED",
            "explanation": "The request is unrelated to LangChain documentation.",
        }

    monkeypatch.setattr(middleware, "_classify_query", _blocked_query)

    asyncio.run(
        middleware.abefore_agent(
            {"messages": [HumanMessage(content="Write a poem about rain.")]},
            Runtime(context=None),
        )
    )

    rejection_content = rejection_model.prompt[1].content
    assert "The request is unrelated to LangChain documentation." in rejection_content


@pytest.mark.parametrize(
    "content",
    [
        "Write a poem about rain.",
        [
            {"type": "text", "text": "Write a poem about rain."},
            {"type": "image_url", "image_url": {"url": "https://example.com/image"}},
        ],
    ],
)
def test_rejection_content_includes_policy_reason(content):
    """Rejection prompts include the policy reason for every content shape."""
    middleware = _middleware_with_models()

    result = middleware._build_rejection_content(
        content, "The request is unrelated to LangChain documentation."
    )

    if isinstance(result, str):
        assert "The request is unrelated to LangChain documentation." in result
    else:
        assert any(
            "The request is unrelated to LangChain documentation." in block.get("text", "")
            for block in result
            if block.get("type") == "text"
        )
