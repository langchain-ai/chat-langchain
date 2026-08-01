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


class ContextAwareModel:
    """Fake classifier that only allows deictic follow-ups when it can see the assistant's turn."""

    def __init__(self):
        self.prompt_text = ""

    def with_structured_output(self, schema):  # noqa: ARG002
        return self

    async def ainvoke(self, prompt, config=None):  # noqa: ARG002
        content = prompt[-1].content
        self.prompt_text = (
            content if isinstance(content, str) else " ".join(str(b) for b in content)
        )
        if "Assistant:" in self.prompt_text:
            return {"decision": "ALLOWED", "explanation": "Follow-up to prior answer."}
        return {"decision": "BLOCKED", "explanation": "No referent for the query."}


def _product_conversation():
    return [
        HumanMessage(content="LangChain 生态里有哪些产品?"),
        AIMessage(
            content=(
                "LangChain is the framework for building LLM apps, LangGraph is the "
                "orchestration runtime, LangSmith is the observability and evaluation "
                "platform, and Deep Agents is the agent harness."
            )
        ),
        HumanMessage(content="有什么区别"),
    ]


def test_guardrails_allows_deictic_followup_using_assistant_context():
    """A short follow-up should be allowed because the assistant's answer is in context."""
    model = ContextAwareModel()
    middleware = _middleware_with_models(("primary", model))

    result = asyncio.run(middleware._classify_query(_product_conversation()))

    assert result["decision"] == "ALLOWED"


def test_guardrails_context_includes_user_and_assistant_turns():
    """The classifier context should be a role-labelled transcript of both sides."""
    model = ContextAwareModel()
    middleware = _middleware_with_models(("primary", model))

    asyncio.run(middleware._classify_query(_product_conversation()))

    assert "Recent conversation:" in model.prompt_text
    assert "User: LangChain 生态里有哪些产品?" in model.prompt_text
    assert "Assistant: LangChain is the framework" in model.prompt_text


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
