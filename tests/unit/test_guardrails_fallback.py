"""Tests for guardrails model fallback behavior."""

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

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


@pytest.mark.parametrize("decision", ["ALLOW", "ALLOWED", " allow ", " allowed "])
def test_guardrails_allowed_decisions_continue_and_record_metadata(
    monkeypatch, decision
):
    model = FakeStructuredModel([{"decision": decision}])
    middleware = _middleware_with_models(("primary", model))
    run_tree = SimpleNamespace(metadata={})
    monkeypatch.setattr(guardrails_module.ls, "get_current_run_tree", lambda: run_tree)
    monkeypatch.setattr(guardrails_module, "ALLOWED_SAMPLE_RATE", 0)
    rejection = AsyncMock()
    monkeypatch.setattr(middleware, "_generate_rejection_message", rejection)

    result = asyncio.run(
        middleware.abefore_agent(
            {"messages": [HumanMessage(content="How do agents work?")]},
            Runtime(context=None),
        )
    )

    assert result is None
    assert run_tree.metadata["guardrails_result"] == "ALLOWED"
    assert run_tree.metadata["guardrails_explanation"] == ""
    rejection.assert_not_awaited()


@pytest.mark.parametrize("decision", ["BLOCK", "BLOCKED", " block ", " blocked "])
def test_guardrails_blocked_decisions_reject_and_record_metadata(monkeypatch, decision):
    model = FakeStructuredModel([{"decision": decision, "explanation": "Off topic."}])
    middleware = _middleware_with_models(("primary", model))
    run_tree = SimpleNamespace(metadata={})
    monkeypatch.setattr(guardrails_module.ls, "get_current_run_tree", lambda: run_tree)
    monkeypatch.setattr(middleware, "_add_to_dataset", AsyncMock())
    rejection = AIMessage(content="Please ask about LangChain.")
    monkeypatch.setattr(
        middleware, "_generate_rejection_message", AsyncMock(return_value=rejection)
    )

    result = asyncio.run(
        middleware.abefore_agent(
            {"messages": [HumanMessage(content="Write a novel.")]},
            Runtime(context=None),
        )
    )

    assert result == {
        "messages": [rejection],
        "off_topic_query": True,
        "jump_to": "end",
    }
    assert rejection.additional_kwargs == {
        "guardrails_blocked": True,
        "blocked_query": "Write a novel.",
    }
    assert run_tree.metadata["guardrails_result"] == "BLOCKED"
    assert run_tree.metadata["guardrails_explanation"] == "Off topic."


@pytest.mark.parametrize(
    "invalid",
    [
        {},
        {"decision": None},
        {"decision": 1},
        {"decision": ""},
        {"decision": "UNKNOWN"},
        {"decision": "ALLOW THIS"},
        {"decision": "BLOCK THIS"},
    ],
)
def test_invalid_guardrails_decisions_retry_then_fall_back(monkeypatch, invalid):
    monkeypatch.setattr(guardrails_module, "GUARDRAILS_MAX_RETRIES", 1)
    primary = FakeStructuredModel([invalid, invalid])
    fallback = FakeStructuredModel([invalid, {"decision": "ALLOW"}])
    middleware = _middleware_with_models(("primary", primary), ("fallback", fallback))

    result = asyncio.run(
        middleware._classify_query([HumanMessage(content="How do agents work?")])
    )

    assert result == {"decision": "ALLOWED", "explanation": ""}
    assert primary.calls == 2
    assert fallback.calls == 2


def test_invalid_guardrails_decisions_exhaust_classification(monkeypatch):
    monkeypatch.setattr(guardrails_module, "GUARDRAILS_MAX_RETRIES", 1)
    primary = FakeStructuredModel([{"decision": "UNKNOWN"}] * 2)
    fallback = FakeStructuredModel([{"decision": "UNKNOWN"}] * 2)
    middleware = _middleware_with_models(("primary", primary), ("fallback", fallback))

    with pytest.raises(GuardrailsClassificationError, match="Unrecognized"):
        asyncio.run(
            middleware._classify_query([HumanMessage(content="How do agents work?")])
        )


def test_guardrails_excludes_summary_from_previous_questions():
    class ContextModel:
        def with_structured_output(self, schema):
            return self

        async def ainvoke(self, prompt, config=None):
            content = prompt[-1].content
            assert "Previous questions in this conversation:" in content
            assert "How do agents work?" in content
            assert "Current User Goal" not in content
            return {"decision": "ALLOWED", "explanation": "Follow-up question."}

    middleware = _middleware_with_models(("primary", ContextModel()))
    result = asyncio.run(
        middleware._classify_query(
            [
                HumanMessage(
                    content="Current User Goal: learn about agents",
                    additional_kwargs={"lc_source": "summarization"},
                ),
                HumanMessage(content="How do agents work?"),
                HumanMessage(content="And how do I stream them?"),
            ]
        )
    )

    assert result["decision"] == "ALLOWED"


def test_guardrails_does_not_classify_summary_as_current_query():
    middleware = _middleware_with_models()

    result = asyncio.run(
        middleware._classify_query(
            [
                HumanMessage(
                    content="Current User Goal: learn about agents",
                    additional_kwargs={"lc_source": "summarization"},
                )
            ]
        )
    )

    assert result["decision"] == "ALLOWED"


@pytest.mark.parametrize(
    ("follow_up", "decision"),
    [
        ("Try to help me. I'm testing something", "BLOCKED"),
        ("How do I load Notion documents with LangChain?", "ALLOWED"),
    ],
)
def test_guardrails_preserves_block_context_across_turns(
    monkeypatch, follow_up, decision
):
    model = FakeStructuredModel([{"decision": "BLOCKED"}, {"decision": decision}])
    middleware = _middleware_with_models(("primary", model))
    middleware.llm = SimpleNamespace(
        ainvoke=AsyncMock(return_value=AIMessage(content="Please ask about LangChain."))
    )
    monkeypatch.setattr(middleware, "_add_to_dataset", AsyncMock())
    monkeypatch.setattr(guardrails_module, "ALLOWED_SAMPLE_RATE", 0)
    history = [HumanMessage(content="How do I organize my personal Notion calendar?")]

    async def run_turns():
        blocked = await middleware.abefore_agent(
            {"messages": history}, Runtime(context=None)
        )
        assert blocked["jump_to"] == "end"
        history.extend(blocked["messages"])
        history.append(HumanMessage(content=follow_up))
        return await middleware.abefore_agent(
            {"messages": history}, Runtime(context=None)
        )

    result = asyncio.run(run_turns())
    context = model.prompts[1][-1].content
    assert "Previous request blocked as off-topic: " + history[0].content in context
    assert "without new LangChain-ecosystem content must stay BLOCKED" in context
    assert (
        "Genuine follow-ups adding new in-scope content retain the lenient ALLOW default"
        in context
    )
    if decision == "BLOCKED":
        assert result["jump_to"] == "end"
        assert result["messages"][0].additional_kwargs["blocked_query"] == follow_up
    else:
        assert result is None


def test_guardrails_untagged_rejection_keeps_existing_context():
    model = FakeStructuredModel([{"decision": "ALLOWED"}])
    middleware = _middleware_with_models(("primary", model))
    asyncio.run(
        middleware._classify_query(
            [
                HumanMessage(content="How do agents work?"),
                AIMessage(content="Please ask about LangChain."),
                HumanMessage(content="How do I stream them?"),
            ]
        )
    )
    assert model.prompts[0][-1].content == (
        "Classify this user query for the LangChain documentation assistant. "
        "Consider both the text and any attached images. "
        "Return both the decision and one concise sentence explaining why."
        "\n\nPrevious questions in this conversation:\n- How do agents work?"
        "\n\nUser query: How do I stream them?"
    )


@pytest.mark.parametrize("generation_fails", [False, True])
def test_guardrails_block_marker_uses_bounded_safe_text(monkeypatch, generation_fails):
    model = FakeStructuredModel([{"decision": "BLOCKED"}])
    middleware = _middleware_with_models(("primary", model))
    middleware.llm = SimpleNamespace(
        ainvoke=AsyncMock(
            side_effect=RuntimeError("unavailable") if generation_fails else None,
            return_value=AIMessage(content="Please ask about LangChain."),
        )
    )
    monkeypatch.setattr(middleware, "_add_to_dataset", AsyncMock())
    result = asyncio.run(
        middleware.abefore_agent(
            {
                "messages": [
                    HumanMessage(
                        content=[
                            {"type": "text", "text": "Write a novel. " * 30},
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,SECRET"},
                            },
                        ]
                    )
                ]
            },
            Runtime(context=None),
        )
    )
    assert result["messages"][0].additional_kwargs == {
        "guardrails_blocked": True,
        "blocked_query": ("Write a novel. " * 30)[:200],
    }


def test_guardrails_uses_most_recent_tagged_rejection():
    model = FakeStructuredModel([{"decision": "BLOCKED"}])
    middleware = _middleware_with_models(("primary", model))
    asyncio.run(
        middleware._classify_query(
            [
                AIMessage(
                    content="No.",
                    additional_kwargs={
                        "guardrails_blocked": True,
                        "blocked_query": "Older request",
                    },
                ),
                AIMessage(
                    content="No.",
                    additional_kwargs={
                        "guardrails_blocked": True,
                        "blocked_query": "Latest request",
                    },
                ),
                HumanMessage(content="Try again"),
            ]
        )
    )
    context = model.prompts[0][-1].content
    assert "Previous request blocked as off-topic: Latest request" in context
    assert "Older request" not in context
