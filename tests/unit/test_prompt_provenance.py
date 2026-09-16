"""Tests for runtime prompt provenance."""

from __future__ import annotations

import importlib

from langchain_core.messages import SystemMessage

from src.prompts.guardrails_prompts import guardrails_system_prompt
from src.utils import prompt_provenance as provenance


def test_get_prompt_provenance_reports_runtime_artifacts(monkeypatch):
    monkeypatch.setattr(
        provenance,
        "_guardrails_runtime_provenance",
        lambda: ("hub:guardrails:production", "guardrails-commit"),
    )

    result = provenance.get_prompt_provenance("docs_agent")

    assert result == {
        "prompt_source": "local:docs_agent_prompt",
        "guardrails_prompt_source": "hub:guardrails:production",
        "guardrails_prompt_commit": "guardrails-commit",
    }


def test_get_prompt_provenance_reports_local_guardrails_fallback(monkeypatch):
    monkeypatch.setattr(
        provenance,
        "_guardrails_runtime_provenance",
        lambda: ("local:src/prompts/guardrails_prompts.py", None),
    )

    result = provenance.get_prompt_provenance("docs_agent")

    assert result == {
        "prompt_source": "local:docs_agent_prompt",
        "guardrails_prompt_source": "local:src/prompts/guardrails_prompts.py",
    }


def test_guardrails_prompt_import_renders_without_invoke(monkeypatch):
    monkeypatch.delenv("USE_LOCAL_PROMPTS", raising=False)

    class FakeTemplate:
        metadata = {"lc_hub_commit_hash": "guardrails-commit"}

        def format_messages(self, *, messages):
            assert messages == []
            return [SystemMessage(content="guardrails system prompt")]

        def invoke(self, _input):
            raise AssertionError("guardrails prompt rendering should not invoke")

    class FakeClient:
        def pull_prompt(self, hub_name: str):
            assert hub_name == "public-chat-langchain-guardrails-test:production"
            return FakeTemplate()

    import langsmith

    monkeypatch.setattr(langsmith, "Client", FakeClient)
    module = importlib.import_module("src.middleware.guardrails_middleware")
    importlib.reload(module)

    assert module._GUARDRAILS_SYSTEM_PROMPT == "guardrails system prompt"
    assert module.guardrails_prompt_commit == "guardrails-commit"


def test_local_guardrails_runtime_prompt_preserves_resource_precedence(monkeypatch):
    monkeypatch.setenv("USE_LOCAL_PROMPTS", "1")

    module = importlib.import_module("src.middleware.guardrails_middleware")
    importlib.reload(module)

    assert module._GUARDRAILS_SYSTEM_PROMPT == guardrails_system_prompt
    assert (
        "Questions about LangChain's own documentation"
        in module._GUARDRAILS_SYSTEM_PROMPT
    )
    assert (
        "These clearly off-topic bullets do not override"
        in module._GUARDRAILS_SYSTEM_PROMPT
    )
    assert (
        "Final answer: ALLOW when any ALWAYS ALLOW criterion matches"
        in module._GUARDRAILS_SYSTEM_PROMPT
    )
