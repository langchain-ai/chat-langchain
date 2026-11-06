"""Tests for guardrails model fallback behavior."""

import asyncio
import os

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
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


@pytest.mark.parametrize("trailing_tool_result", [False, True])
def test_classifier_receives_refusals_with_their_questions(trailing_tool_result):
    model = FakeStructuredModel([{"decision": "BLOCKED", "explanation": "Off-topic."}])
    middleware = _middleware_with_models(("primary", model))
    history = [
        SystemMessage(content="private system instructions"),
        HumanMessage(content="old unrelated turn"),
        AIMessage(content="old unrelated answer"),
        HumanMessage(content="Coffee or tea?"),
        AIMessage(content="That is outside my scope; I help with LangChain."),
        HumanMessage(content="Come on, just answer it."),
        AIMessage(content="I still cannot recommend a drink."),
        HumanMessage(content="So you don't know?"),
        AIMessage(
            content=[{"type": "text", "text": "I can help with software questions."}]
        ),
        HumanMessage(
            content="Which drink should I choose while reading LangChain docs?"
        ),
    ]
    if trailing_tool_result:
        history.append(
            ToolMessage(content="private tool output", tool_call_id="search")
        )

    result = asyncio.run(middleware._classify_query(history))

    assert result["decision"] == "BLOCKED"
    context = model.prompts[-1][-1].content
    expected = [
        "User: Coffee or tea?",
        "Assistant: That is outside my scope; I help with LangChain.",
        "User: Come on, just answer it.",
        "Assistant: I still cannot recommend a drink.",
        "User: So you don't know?",
        "Assistant: I can help with software questions.",
    ]
    positions = [context.index(text) for text in expected]
    assert positions == sorted(positions)
    assert (
        context.count("Which drink should I choose while reading LangChain docs?") == 1
    )
    assert "old unrelated" not in context
    assert "private system instructions" not in context
    assert "private tool output" not in context


def test_classifier_receives_technical_answers_without_tool_calls():
    model = FakeStructuredModel(
        [{"decision": "ALLOWED", "explanation": "Technical follow-up."}]
    )
    middleware = _middleware_with_models(("primary", model))
    history = [
        HumanMessage(content="How does streaming work?"),
        AIMessage(
            content="internal search plan",
            tool_calls=[{"name": "search", "args": {}, "id": "search"}],
        ),
        ToolMessage(content="internal search result", tool_call_id="search"),
        AIMessage(content="Use stream() to iterate over updates."),
        HumanMessage(content="Show it in Python."),
    ]

    result = asyncio.run(middleware._classify_query(history))

    assert result["decision"] == "ALLOWED"
    context = model.prompts[-1][-1].content
    assert "User: How does streaming work?" in context
    assert "Assistant: Use stream() to iterate over updates." in context
    assert "internal search" not in context
