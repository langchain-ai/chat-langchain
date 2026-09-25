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


def test_guardrails_blocks_rephrased_query_after_prior_refusal(monkeypatch):
    """A rephrased version of a refused topic remains blocked."""
    classifier = FakeStructuredModel(
        [
            {"decision": "BLOCKED", "explanation": "Off-topic."},
            {"decision": "BLOCKED", "explanation": "Prior refusal."},
        ]
    )
    middleware = _middleware_with_models(
        (
            "primary",
            classifier,
        )
    )

    async def _rejection_message(content):  # noqa: ARG001
        return AIMessage(content="That topic is outside my scope.")

    monkeypatch.setattr(middleware, "_generate_rejection_message", _rejection_message)
    first_query = HumanMessage(content="How should we handle QHSE compliance?")
    first_result = asyncio.run(
        middleware.abefore_agent(
            {"messages": [first_query]},
            Runtime(context=None),
        )
    )

    second_query = HumanMessage(content="How can our agents integrate this?")
    second_state = {
        "messages": [first_query, *first_result["messages"], second_query],
        "blocked_queries": first_result["blocked_queries"],
    }
    second_result = asyncio.run(
        middleware.abefore_agent(second_state, Runtime(context=None))
    )

    assert second_result["off_topic_query"] is True
    assert second_result["jump_to"] == "end"
    second_prompt = classifier.prompts[1][1].content
    assert "Previous turns in this conversation (refusals are marked)" in second_prompt
    assert "User [REFUSED]: How should we handle QHSE compliance?" in second_prompt
    assert "Previously refused queries:" in second_prompt


def test_guardrails_allowed_turn_resets_off_topic_query():
    """An allowed turn clears the prior off-topic flag."""
    middleware = _middleware_with_models(
        (
            "primary",
            FakeStructuredModel([{"decision": "ALLOWED", "explanation": "Technical."}]),
        )
    )

    result = asyncio.run(
        middleware.abefore_agent(
            {
                "messages": [HumanMessage(content="How do agents work?")],
                "off_topic_query": True,
            },
            Runtime(context=None),
        )
    )

    assert result == {"off_topic_query": False}


def test_guardrails_allows_technical_follow_up_after_allowed_turn():
    """An in-scope technical follow-up remains allowed."""
    middleware = _middleware_with_models(
        (
            "primary",
            FakeStructuredModel(
                [{"decision": "ALLOWED", "explanation": "Technical follow-up."}]
            ),
        )
    )

    result = asyncio.run(
        middleware.abefore_agent(
            {
                "messages": [
                    HumanMessage(content="How do agents work?"),
                    AIMessage(content="Agents can call tools."),
                    HumanMessage(content="Can you show that in Python with LangChain?"),
                ],
                "off_topic_query": False,
            },
            Runtime(context=None),
        )
    )

    assert result == {"off_topic_query": False}
