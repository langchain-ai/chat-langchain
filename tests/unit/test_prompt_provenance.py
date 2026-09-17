"""Tests for Hub prompt provenance resolution."""

from __future__ import annotations

import importlib

from langchain_core.messages import SystemMessage

from src.utils import prompt_provenance as provenance


class _FakeTemplate:
    def __init__(self, commit: str | None):
        self.metadata = {"lc_hub_commit_hash": commit} if commit else {}


def test_get_prompt_provenance_local_mode(monkeypatch):
    monkeypatch.setattr(provenance, "_USE_LOCAL_PROMPTS", True)

    result = provenance.get_prompt_provenance("docs_agent")
    assert result == {
        "prompt_source": "local:instructions.md",
        "guardrails_prompt_source": "local:src/prompts/guardrails_prompts.py",
        "guardrails_prompt_matches_repo": True,
        "guardrails_prompt_sha256": provenance.hashlib.sha256(
            provenance.guardrails_system_prompt.encode()
        ).hexdigest(),
    }
    assert "prompt_commit" not in result


def test_resolve_hub_provenance_uses_prompt_workspace_and_api_key(monkeypatch):
    provenance._resolve_hub_provenance.cache_clear()
    provenance._resolve_guardrails_provenance.cache_clear()
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

    assert len(constructed) == 2
    assert all(
        call.get("workspace_id") == "ebbaf2eb-769b-4505-aca2-d11de10372a4"
        and call.get("api_key") == "lsv2_prompt_test_key"
        for call in constructed
    )
    assert result["prompt_commit"] == (
        "commit-for-public-chat-langchain-test:production"
    )
    assert result["guardrails_prompt_commit"] == (
        "commit-for-public-chat-langchain-guardrails-test:production"
    )


def test_resolve_hub_provenance_without_overrides_uses_default_client(monkeypatch):
    provenance._resolve_hub_provenance.cache_clear()
    provenance._resolve_guardrails_provenance.cache_clear()
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

    assert constructed == [{}, {}]
    assert result["prompt_source"].startswith("hub:")
    assert "prompt_commit" not in result


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

    assert module._GUARDRAILS_SYSTEM_PROMPT == module._LOCAL_GUARDRAILS_SYSTEM_PROMPT
    assert module.guardrails_prompt_commit == "guardrails-commit"
    assert module.guardrails_prompt_matches_repo is False


def test_guardrails_prompt_import_falls_back_when_hub_text_differs(monkeypatch):
    monkeypatch.delenv("USE_LOCAL_PROMPTS", raising=False)

    class FakeTemplate:
        metadata = {"lc_hub_commit_hash": "stale-commit"}

        def format_messages(self, *, messages):
            return [SystemMessage(content="stale guardrails system prompt")]

    class FakeClient:
        def pull_prompt(self, hub_name: str):
            return FakeTemplate()

    import langsmith

    monkeypatch.setattr(langsmith, "Client", FakeClient)
    module = importlib.import_module("src.middleware.guardrails_middleware")
    importlib.reload(module)

    assert module._GUARDRAILS_SYSTEM_PROMPT == module._LOCAL_GUARDRAILS_SYSTEM_PROMPT
    assert module.guardrails_prompt_commit == "stale-commit"
    assert module.guardrails_prompt_matches_repo is False
    assert module.guardrails_prompt_sha256 == module.hashlib.sha256(
        b"stale guardrails system prompt"
    ).hexdigest()


def test_guardrails_prompt_provenance_records_content_hash(monkeypatch):
    provenance._resolve_guardrails_provenance.cache_clear()
    monkeypatch.setattr(provenance, "_USE_LOCAL_PROMPTS", False)
    monkeypatch.delenv("LANGSMITH_PROMPT_WORKSPACE_ID", raising=False)
    monkeypatch.delenv("LANGSMITH_PROMPT_API_KEY", raising=False)

    class FakeTemplate:
        metadata = {"lc_hub_commit_hash": "stale-commit"}

        def format_messages(self, *, messages):
            return [SystemMessage(content="stale guardrails system prompt")]

    class FakeClient:
        def pull_prompt(self, hub_name: str):
            return FakeTemplate()

    import langsmith

    monkeypatch.setattr(langsmith, "Client", FakeClient)
    result = provenance.get_prompt_provenance("docs_agent")

    assert result["guardrails_prompt_matches_repo"] is False
    assert result["guardrails_prompt_sha256"] == provenance.hashlib.sha256(
        b"stale guardrails system prompt"
    ).hexdigest()
