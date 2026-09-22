"""Tests for guardrails model fallback behavior."""

import asyncio
import os

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
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


def test_guardrails_hub_bootstrap_does_not_trace_prompt_render(monkeypatch):
    """Hub prompt bootstrap renders without creating a traced run."""
    run_creations = []
    invoke_calls = []

    class FakePrompt:
        metadata = {"lc_hub_commit_hash": "hub-revision"}

        def format_messages(self, **kwargs):  # noqa: ARG002
            return [SystemMessage(content="hub prompt")]

        def invoke(self, *args, **kwargs):  # noqa: ARG002
            invoke_calls.append(True)
            raise AssertionError("prompt bootstrap must not invoke the Runnable")

    class FakeClient:
        def pull_prompt(self, name):  # noqa: ARG002
            return FakePrompt()

    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setattr(guardrails_module, "_USE_LOCAL_PROMPTS", False)
    monkeypatch.setattr(guardrails_module, "Client", FakeClient)

    class RecordingClient(FakeClient):
        def create_run(self, *args, **kwargs):  # noqa: ARG002
            run_creations.append(True)

    monkeypatch.setattr(guardrails_module, "Client", RecordingClient)
    guardrails_module._load_guardrails_prompt()

    assert guardrails_module._GUARDRAILS_SYSTEM_PROMPT == "hub prompt"
    assert guardrails_module.guardrails_prompt_commit == "hub-revision"
    assert run_creations == []
    assert invoke_calls == []


def test_guardrails_hub_bootstrap_falls_back_to_local_prompt(monkeypatch):
    """Hub prompt bootstrap preserves the local fallback on pull failure."""

    class FailingClient:
        def pull_prompt(self, name):  # noqa: ARG002
            raise RuntimeError("hub unavailable")

    monkeypatch.setattr(guardrails_module, "_USE_LOCAL_PROMPTS", False)
    monkeypatch.setattr(guardrails_module, "Client", FailingClient)

    guardrails_module._load_guardrails_prompt()

    assert (
        guardrails_module._GUARDRAILS_SYSTEM_PROMPT
        == guardrails_module._LOCAL_GUARDRAILS_SYSTEM_PROMPT
    )
    assert guardrails_module.guardrails_prompt_commit is None
    assert (
        guardrails_module.guardrails_prompt_source
        == "local:src/prompts/guardrails_prompts.py"
    )


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
