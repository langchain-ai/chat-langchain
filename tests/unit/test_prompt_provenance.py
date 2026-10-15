"""Tests for Hub prompt publication and provenance resolution."""

from __future__ import annotations

from pathlib import Path

from scripts import push_docs_agent_prompt
from src.utils import prompt_provenance as provenance


def test_published_docs_prompt_public_facts_match_managed_instructions(monkeypatch):
    published = {}

    class FakeClient:
        def push_prompt(self, identifier, *, object):
            published[identifier] = object
            return "https://smith.langchain.com/prompts/test"

    monkeypatch.setattr(push_docs_agent_prompt, "Client", FakeClient)
    monkeypatch.setattr(
        push_docs_agent_prompt, "load_dotenv", lambda *args, **kwargs: None
    )

    push_docs_agent_prompt.main()

    template = published["langchain-ai/public-chat-langchain-test"]
    messages = template.invoke({}).messages
    instructions = Path(__file__).resolve().parents[2] / "instructions.md"
    assert len(messages) == 1
    assert messages[0].type == "system"
    section = "## About Chat LangChain (public facts you may share)"
    published_facts = (
        messages[0].content.split(section)[1].split("## Available Tools")[0]
    )
    managed_facts = (
        instructions.read_text().split(section)[1].split("## Available Tools")[0]
    )
    assert published_facts == managed_facts
    assert "https://github.com/langchain-ai/chat-langchain" in messages[0].content
    assert "PYLON_KB_ID" in messages[0].content
    assert "managed-deepagents" in messages[0].content


class _FakeTemplate:
    def __init__(self, commit: str | None):
        self.metadata = {"lc_hub_commit_hash": commit} if commit else {}


def test_get_prompt_provenance_local_mode(monkeypatch):
    monkeypatch.setattr(provenance, "_USE_LOCAL_PROMPTS", True)

    result = provenance.get_prompt_provenance("docs_agent")
    assert result == {
        "prompt_source": "local:instructions.md",
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
