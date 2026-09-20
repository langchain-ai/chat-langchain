"""Tests for guardrails model fallback behavior."""

import asyncio
import os

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
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


def test_blocked_turn_does_not_poison_later_docs_search(monkeypatch):
    """A middleware refusal is hidden from a later in-scope model call."""
    middleware = _middleware_with_models()
    decisions = iter(
        [
            {"decision": "BLOCKED", "explanation": "Off topic."},
            {"decision": "ALLOWED", "explanation": "LangChain question."},
        ]
    )

    async def classify(messages, guardrail_history=None):  # noqa: ARG001
        return next(decisions)

    async def rejection(content):  # noqa: ARG001
        return AIMessage(
            content="I can only help with LangChain questions.",
            response_metadata={"guardrail_refusal": True},
        )

    monkeypatch.setattr(middleware, "_classify_query", classify)
    monkeypatch.setattr(middleware, "_generate_rejection_message", rejection)

    blocked_state = {"messages": [HumanMessage(content="Write a poem.")]}
    blocked_update = asyncio.run(
        middleware.abefore_agent(blocked_state, Runtime(context=None))
    )
    state = {
        "messages": [*blocked_state["messages"], *blocked_update["messages"]],
        "guardrail_history": blocked_update["guardrail_history"],
    }
    state["messages"].append(
        HumanMessage(content="How do I configure a LangGraph checkpointer?")
    )

    allowed_update = asyncio.run(
        middleware.abefore_agent(state, Runtime(context=None))
    )
    assert "messages" not in allowed_update

    calls = []

    async def handler(request):
        calls.append(request)
        return ModelResponse(
            result=[
                AIMessage(
                    content="I’m searching the documentation now.",
                    tool_calls=[
                        {
                            "name": "search_docs_by_lang_chain",
                            "args": {"query": "LangGraph checkpointer"},
                            "id": "search-1",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        )

    response = asyncio.run(
        middleware.awrap_model_call(
            ModelRequest(model=object(), messages=state["messages"]), handler
        )
    )

    assert len(calls) == 1
    assert not any(
        message.response_metadata.get("guardrail_refusal") is True
        for message in calls[0].messages
        if isinstance(message, AIMessage)
    )
    assert response.result[0].tool_calls[0]["name"] == "search_docs_by_lang_chain"
