"""Tests for guardrails model fallback behavior."""

import asyncio
import importlib
import os

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
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


@pytest.mark.parametrize(
    "answer",
    [
        "LangSmith migration " * 30,
        [{"type": "text", "text": "LangSmith migration " * 30}],
    ],
)
def test_guardrails_includes_recent_assistant_text_and_prior_queries(answer):
    model = FakeStructuredModel([{"decision": "ALLOWED", "explanation": "Follow-up."}])
    middleware = _middleware_with_models(("primary", model))
    messages = [HumanMessage(content="old question"), AIMessage(content="old answer")]
    questions = [f"question {i} " + "x" * 250 for i in range(3)]
    messages.extend(HumanMessage(content=q) for q in questions)
    messages.extend(
        [
            AIMessage(content=answer),
            AIMessage(
                content="", tool_calls=[{"name": "search", "args": {}, "id": "search"}]
            ),
            ToolMessage(content="retrieved docs", tool_call_id="search"),
            HumanMessage(content="in English"),
        ]
    )

    result = asyncio.run(middleware._classify_query(messages))

    assert result["decision"] == "ALLOWED"
    content = model.prompts[0][1].content
    expected_questions = "\n".join(f"- {q[:200]}" for q in questions)
    assert f"Previous questions in this conversation:\n{expected_questions}" in content
    assert "old question" not in content
    assert "old answer" not in content
    assert "retrieved docs" not in content
    answer_context = content.split("Previous assistant answer:\n", 1)[1]
    assert (
        answer_context
        == ("LangSmith migration " * 30)[:300] + "\n\nUser query: in English"
    )


@pytest.mark.parametrize("assistant_messages", [[], [AIMessage(content="")]])
def test_guardrails_omits_assistant_context_without_text(assistant_messages):
    model = FakeStructuredModel([{"decision": "ALLOWED", "explanation": "Technical."}])
    middleware = _middleware_with_models(("primary", model))
    messages = [
        HumanMessage(content="LangGraph SDK"),
        *assistant_messages,
        HumanMessage(content="How do I migrate regions?"),
    ]

    asyncio.run(middleware._classify_query(messages))

    content = model.prompts[0][1].content
    assert "Previous assistant answer:" not in content
    assert "Previous questions in this conversation:\n- LangGraph SDK" in content


@pytest.mark.parametrize(
    "local_prompts,environment", [("", ""), ("", "dev"), ("1", "")]
)
def test_guardrails_uses_bundled_policy_when_hub_has_stale_rules(
    monkeypatch, local_prompts, environment
):
    from langchain_core.prompts import ChatPromptTemplate

    from src.prompts.guardrails_prompts import guardrails_system_prompt

    class StaleHubClient:
        def pull_prompt(self, name):
            return ChatPromptTemplate.from_messages(
                [("system", "Block all safety evaluation and unknown product terms.")]
            )

    monkeypatch.setenv("USE_LOCAL_PROMPTS", local_prompts)
    monkeypatch.setenv("LANGSMITH_ENV", environment)
    monkeypatch.setattr("langsmith.Client", StaleHubClient)
    importlib.reload(guardrails_module)
    model = FakeStructuredModel(
        [{"decision": "ALLOWED", "explanation": "Safety evaluation."}]
    )
    middleware = _middleware_with_models(("primary", model))

    asyncio.run(
        middleware._classify_query(
            [HumanMessage(content="How do I evaluate whether my agent refuses fraud?")]
        )
    )

    assert model.prompts[0][0].content == guardrails_system_prompt
