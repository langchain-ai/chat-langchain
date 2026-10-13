"""Tests for Hub prompt provenance resolution."""

from __future__ import annotations

import importlib

import pytest
from langchain_core.messages import SystemMessage

from src.prompts.guardrails_prompts import guardrails_system_prompt
from src.utils import prompt_provenance as provenance


class _FakeTemplate:
    def __init__(self, commit: str | None):
        self.metadata = {"lc_hub_commit_hash": commit} if commit else {}

    def format_messages(self, *, messages):
        assert messages == []
        return [SystemMessage(content=guardrails_system_prompt)]


@pytest.fixture
def isolated_guardrails_module():
    module = importlib.import_module("src.middleware.guardrails_middleware")
    original = vars(module).copy()
    yield module
    vars(module).clear()
    vars(module).update(original)


def test_get_prompt_provenance_local_mode(monkeypatch):
    monkeypatch.setattr(provenance, "_USE_LOCAL_PROMPTS", True)

    result = provenance.get_prompt_provenance("docs_agent")
    assert result == {
        "prompt_source": "context_hub:/instructions.md",
        "guardrails_prompt_source": "local:src/prompts/guardrails_prompts.py",
    }
    assert "prompt_commit" not in result


def test_resolve_hub_provenance_uses_prompt_workspace_and_api_key(monkeypatch):
    provenance._resolve_hub_provenance.cache_clear()
    monkeypatch.setattr(provenance, "_USE_LOCAL_PROMPTS", False)
    monkeypatch.setenv(
        "LANGSMITH_PROMPT_WORKSPACE_ID", "ebbaf2eb-769b-4505-aca2-d11de10372a4"
    )
    monkeypatch.setenv("LANGSMITH_PROMPT_API_KEY", "lsv2_prompt_test_key")

    constructed: list[dict[str, object]] = []

    class FakeClient:
        def __init__(self, *args, **kwargs):
            constructed.append(kwargs)

        def pull_prompt(self, hub_name: str):
            return _FakeTemplate(f"commit-for-{hub_name}")

    import langsmith

    monkeypatch.setattr(langsmith, "Client", FakeClient)

    result = provenance.get_prompt_provenance("docs_agent")

    assert len(constructed) == 1
    assert all(
        call.get("workspace_id") == "ebbaf2eb-769b-4505-aca2-d11de10372a4"
        and call.get("api_key") == "lsv2_prompt_test_key"
        for call in constructed
    )
    assert result["prompt_source"] == "context_hub:/instructions.md"
    assert "prompt_commit" not in result
    assert result["guardrails_prompt_commit"] == (
        "commit-for-public-chat-langchain-guardrails-test:production"
    )


def test_resolve_hub_provenance_without_overrides_uses_default_client(monkeypatch):
    provenance._resolve_hub_provenance.cache_clear()
    monkeypatch.setattr(provenance, "_USE_LOCAL_PROMPTS", False)
    monkeypatch.delenv("LANGSMITH_PROMPT_WORKSPACE_ID", raising=False)
    monkeypatch.delenv("LANGSMITH_PROMPT_API_KEY", raising=False)

    constructed: list[dict[str, object]] = []

    class FakeClient:
        def __init__(self, *args, **kwargs):
            constructed.append(kwargs)

        def pull_prompt(self, hub_name: str):
            return _FakeTemplate(None)

    import langsmith

    monkeypatch.setattr(langsmith, "Client", FakeClient)

    result = provenance.get_prompt_provenance("docs_agent")

    assert constructed == [{}]
    assert result["prompt_source"] == "context_hub:/instructions.md"
    assert "prompt_commit" not in result


@pytest.mark.parametrize("drift", [False, True])
def test_guardrails_prompt_import_renders_without_invoke(
    monkeypatch, drift, isolated_guardrails_module
):
    monkeypatch.delenv("USE_LOCAL_PROMPTS", raising=False)

    class FakeTemplate:
        metadata = {"lc_hub_commit_hash": "guardrails-commit"}

        def format_messages(self, *, messages):
            assert messages == []
            content = (
                "stale guardrails system prompt" if drift else guardrails_system_prompt
            )
            return [SystemMessage(content=content)]

        def invoke(self, _input):
            raise AssertionError("guardrails prompt rendering should not invoke")

    class FakeClient:
        def pull_prompt(self, hub_name: str):
            assert hub_name == "public-chat-langchain-guardrails-test:production"
            return FakeTemplate()

    import langsmith

    monkeypatch.setattr(langsmith, "Client", FakeClient)
    module = isolated_guardrails_module
    importlib.reload(module)

    assert module._GUARDRAILS_SYSTEM_PROMPT == guardrails_system_prompt
    assert module.guardrails_prompt_commit == (None if drift else "guardrails-commit")
    expected_source = (
        "local:src/prompts/guardrails_prompts.py"
        if drift
        else "hub:public-chat-langchain-guardrails-test:production"
    )
    assert module.guardrails_prompt_source == expected_source


def test_drifted_hub_provenance_reports_local_policy(monkeypatch):
    provenance._resolve_hub_provenance.cache_clear()
    monkeypatch.setattr(provenance, "_USE_LOCAL_PROMPTS", False)

    class FakeClient:
        def pull_prompt(self, hub_name):
            template = _FakeTemplate("stale-commit")
            template.format_messages = lambda **kwargs: [SystemMessage(content="stale")]
            return template

    monkeypatch.setattr(provenance, "_hub_client", lambda *args: FakeClient())
    assert provenance.get_prompt_provenance("docs_agent") == {
        "prompt_source": "context_hub:/instructions.md",
        "guardrails_prompt_source": "local:src/prompts/guardrails_prompts.py",
    }
    provenance._resolve_hub_provenance.cache_clear()
