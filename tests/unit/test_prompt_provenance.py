"""Tests for Hub prompt provenance resolution."""

from __future__ import annotations

from src.utils import prompt_provenance as provenance


class _FakeTemplate:
    def __init__(self, commit: str | None):
        self.metadata = {"lc_hub_commit_hash": commit} if commit else {}


def test_get_prompt_provenance_local_mode(monkeypatch):
    monkeypatch.setattr(provenance, "_USE_LOCAL_PROMPTS", True)

    result = provenance.get_prompt_provenance("docs_agent")
    assert result == {
        "prompt_source": "local:src/prompts/docs_agent_prompt.py",
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
