"""Tests for ingress guards input caps and root-trace metadata helpers."""

from __future__ import annotations

import os
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

os.environ["USE_LOCAL_PROMPTS"] = "1"

from src.middleware.ingress_guards_middleware import (
    MAX_MESSAGE_CHARS,
    IngressGuardsMiddleware,
    redact_secrets,
)
from src.utils.trace_root_metadata import build_docs_agent_trace_metadata


def test_before_agent_truncates_oversized_human_message():
    middleware = IngressGuardsMiddleware()
    long_text = "x" * (MAX_MESSAGE_CHARS + 50)
    human = HumanMessage(content=long_text, id="h1")
    state = {"messages": [AIMessage(content="hi"), human]}

    update = middleware.before_agent(state, runtime=SimpleNamespace())

    assert update is not None
    assert update["messages"][0].id == "h1"
    assert len(update["messages"][0].content) == MAX_MESSAGE_CHARS


def test_before_agent_noop_when_under_cap():
    middleware = IngressGuardsMiddleware()
    state = {"messages": [HumanMessage(content="Hello", id="h1")]}

    assert middleware.before_agent(state, runtime=SimpleNamespace()) is None


def test_redact_secrets_preserves_uri_structure():
    text = "postgres://user:supersecret@host:5432/db?sslmode=disable"

    assert (
        redact_secrets(text)
        == "postgres://user:[REDACTED_SECRET]@host:5432/db?sslmode=disable"
    )


def test_redact_secrets_leaves_documentation_sample_untouched():
    text = "postgresql://usuario:password@localhost/db"

    assert redact_secrets(text) == text


def test_redact_secrets_redacts_key_in_fenced_code():
    text = "```bash\nexport KEY=sk-live-secret-value\n```"

    assert redact_secrets(text) == "```bash\nexport KEY=[REDACTED_SECRET]\n```"


def test_redact_secrets_leaves_environment_lookup_untouched():
    text = 'api_key = os.getenv("X")'

    assert redact_secrets(text) == text


def test_before_agent_preserves_identity_for_clean_message():
    middleware = IngressGuardsMiddleware()
    human = HumanMessage(content="Hello", id="h1")
    state = {"messages": [human]}

    assert middleware.before_agent(state, runtime=SimpleNamespace()) is None
    assert state["messages"][0] is human


def test_build_docs_agent_trace_metadata_includes_provenance_and_version(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_REVISION_ID", "rev-a")
    monkeypatch.setenv("LANGSMITH_HOST_REVISION_ID", "rev-b")
    monkeypatch.setattr(
        "src.utils.prompt_provenance._USE_LOCAL_PROMPTS",
        True,
    )

    metadata = build_docs_agent_trace_metadata()

    assert metadata["source_type"] == "Chat-LangChain"
    assert metadata["prompt_source"] == "local:instructions.md"
    assert (
        metadata["guardrails_prompt_source"]
        == "local:src/prompts/guardrails_prompts.py"
    )
    assert metadata["LANGSMITH_AGENT_VERSION"] == "rev-a"


def test_build_docs_agent_trace_metadata_falls_back_to_host_revision(monkeypatch):
    monkeypatch.delenv("LANGCHAIN_REVISION_ID", raising=False)
    monkeypatch.setenv("LANGSMITH_HOST_REVISION_ID", "host-rev")

    metadata = build_docs_agent_trace_metadata()
    assert metadata["LANGSMITH_AGENT_VERSION"] == "host-rev"
