"""Tests for guardrails model fallback behavior."""

import asyncio
import os

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
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

    assert result == {"off_topic_query": False, "guardrails_decision": None}


def test_guardrails_allowed_turn_returns_verdict_and_clears_previous_block(monkeypatch):
    decision = {"decision": "ALLOWED", "explanation": "An in-scope follow-up."}
    middleware = _middleware_with_models(("primary", FakeStructuredModel([decision])))
    history = [{"query": "Write a story.", "decision": "BLOCKED"}]
    monkeypatch.setattr(guardrails_module, "ALLOWED_SAMPLE_RATE", 0)

    result = asyncio.run(
        middleware.abefore_agent(
            {
                "messages": [HumanMessage(content="How do LangGraph agents work?")],
                "off_topic_query": True,
                "guardrails_decision": {
                    "decision": "BLOCKED",
                    "explanation": "Fiction.",
                },
                "guardrail_history": history,
            },
            Runtime(context=None),
        )
    )

    assert result == {
        "off_topic_query": False,
        "guardrails_decision": decision,
        "guardrail_history": [
            *history,
            {"query": "How do LangGraph agents work?", "decision": "ALLOWED"},
        ],
    }
    assert history == [{"query": "Write a story.", "decision": "BLOCKED"}]


def test_guardrails_failed_classification_clears_stale_verdict_and_history(monkeypatch):
    middleware = _middleware_with_models()

    async def _raise_classification_error(messages, guardrail_history):
        raise GuardrailsClassificationError("all models failed")

    monkeypatch.setattr(middleware, "_classify_query", _raise_classification_error)
    result = asyncio.run(
        middleware.abefore_agent(
            {
                "messages": [HumanMessage(content="How do agents work?")],
                "guardrails_decision": {
                    "decision": "BLOCKED",
                    "explanation": "Off-topic.",
                },
                "guardrail_history": [
                    {"query": "Write a story.", "decision": "BLOCKED"}
                ],
            },
            Runtime(context=None),
        )
    )

    assert result == {
        "off_topic_query": False,
        "guardrails_decision": None,
        "guardrail_history": [],
    }


@pytest.mark.parametrize(
    "system_message",
    [
        None,
        SystemMessage(content="Docs instructions."),
        SystemMessage(content=[{"type": "text", "text": "Docs instructions."}]),
    ],
)
def test_guardrails_verdict_reaches_model_context_without_changing_history(
    system_message,
):
    middleware = _middleware_with_models()
    request = ModelRequest(
        model=object(),
        messages=[HumanMessage(content="How do agents work?")],
        system_message=system_message,
        state={
            "guardrails_decision": {"decision": "ALLOWED", "explanation": "In scope."}
        },
    )
    response = ModelResponse(result=[AIMessage(content="Which kind of agent?")])
    calls = []

    async def handler(request):
        calls.append(request)
        return response

    result = asyncio.run(middleware.awrap_model_call(request, handler))

    assert result is response
    assert len(calls) == 1
    assert "Current-turn guardrails verdict: ALLOWED." in calls[0].system_message.text
    if system_message is not None:
        assert "Docs instructions." in calls[0].system_message.text
    assert calls[0].messages == request.messages
    assert request.system_message is system_message


@pytest.mark.parametrize("state", [{}, {"guardrails_decision": None}])
def test_guardrails_without_verdict_leaves_model_request_unchanged(state):
    middleware = _middleware_with_models()
    request = ModelRequest(model=object(), messages=[], state=state)

    async def handler(actual_request):
        assert actual_request is request
        return ModelResponse(result=[AIMessage(content="Hello.")])

    asyncio.run(middleware.awrap_model_call(request, handler))
