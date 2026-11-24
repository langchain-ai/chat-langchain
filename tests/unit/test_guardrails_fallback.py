"""Tests for guardrails model fallback behavior."""

import asyncio
import os
import runpy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import langsmith as ls
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tracers.langchain import wait_for_all_tracers
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


@pytest.mark.parametrize("commit", [None, "guardrails-hub-commit"])
def test_hub_prompt_load_preserves_text_without_tracing(monkeypatch, commit):
    monkeypatch.setenv("USE_LOCAL_PROMPTS", "0")
    template = ChatPromptTemplate.from_messages(
        [("system", "Guardrails {scope}"), ("placeholder", "{messages}")]
    ).partial(scope="system text")
    template.metadata = {"lc_hub_commit_hash": commit} if commit else {}
    pull = Mock(return_value=template)
    monkeypatch.setattr(ls.Client, "pull_prompt", pull)
    client = Mock(spec=ls.Client)

    with ls.tracing_context(enabled=True, client=client):
        loaded = runpy.run_path(guardrails_module.__file__)
        wait_for_all_tracers()
        client.create_run.assert_not_called()
        assert loaded["_GUARDRAILS_SYSTEM_PROMPT"] == "Guardrails system text"
        assert loaded["guardrails_prompt_commit"] == commit
        assert loaded["guardrails_prompt_source"] == (
            f"hub:{loaded['_GUARDRAILS_PROMPT_HUB_NAME']}"
        )
        pull.assert_called_once_with(loaded["_GUARDRAILS_PROMPT_HUB_NAME"])

        result = template.invoke({"messages": []})
        wait_for_all_tracers()
        assert result.messages[0].content == loaded["_GUARDRAILS_SYSTEM_PROMPT"]
        client.create_run.assert_called_once()


@pytest.mark.parametrize("local_mode", [False, True])
def test_prompt_load_local_fallback_preserves_provenance(monkeypatch, local_mode):
    monkeypatch.setenv("USE_LOCAL_PROMPTS", "1" if local_mode else "0")
    pull = Mock(side_effect=RuntimeError("Hub unavailable"))
    monkeypatch.setattr(ls.Client, "pull_prompt", pull)
    client = Mock(spec=ls.Client)

    with ls.tracing_context(enabled=True, client=client):
        loaded = runpy.run_path(guardrails_module.__file__)
        wait_for_all_tracers()

    assert loaded["_GUARDRAILS_SYSTEM_PROMPT"] == (
        guardrails_module._LOCAL_GUARDRAILS_SYSTEM_PROMPT
    )
    assert loaded["guardrails_prompt_commit"] is None
    assert loaded["guardrails_prompt_source"] == (
        "local:src/prompts/guardrails_prompts.py"
    )
    assert pull.call_count == (0 if local_mode else 1)
    client.create_run.assert_not_called()
