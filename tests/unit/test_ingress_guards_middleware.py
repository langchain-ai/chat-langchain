"""Tests for ingress guards input caps and root-trace metadata helpers."""

from __future__ import annotations

import os
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

os.environ["USE_LOCAL_PROMPTS"] = "1"

from src.middleware.ingress_guards_middleware import (
    MAX_MESSAGE_CHARS,
    IngressGuardsMiddleware,
    _redact_secrets,
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


def test_redact_secrets_in_curl_api_key():
    text = 'curl -H "X-Api-Key: lsv2_sk_1234567890abcdef" https://example.com'

    assert "lsv2_sk_" not in _redact_secrets(text)
    assert "<REDACTED_API_KEY>" in _redact_secrets(text)


def test_redact_assignment_uri_and_path_identifiers():
    text = (
        "password=correct-horse-battery-staple "
        "postgresql://alice:correct-horse-battery-staple@db.internal/app "
        "/home/alice@corp.internal/project"
    )

    redacted = _redact_secrets(text)

    assert "correct-horse-battery-staple" not in redacted
    assert redacted.count("<REDACTED_PASSWORD>") == 2
    assert "<REDACTED_PERSONAL_IDENTIFIER>" in redacted


def test_redact_secrets_preserves_placeholders():
    text = "password=YOUR_PASSWORD_HERE key=sk-example-xxxxxxxxxxxxxxxxxxxx"

    assert _redact_secrets(text) == text


def test_before_agent_sanitizes_text_blocks_and_preserves_truncation_cap():
    middleware = IngressGuardsMiddleware()
    content = [
        {"type": "text", "text": "token=ghp_123456789012345678901234567890"},
        "x" * MAX_MESSAGE_CHARS,
    ]
    human = HumanMessage(content=content, id="h1")

    update = middleware.before_agent({"messages": [human]}, runtime=SimpleNamespace())

    assert update is not None
    assert update["messages"][0].content[0]["text"] == "token=<REDACTED_TOKEN>"
    assert len(update["messages"][0].content[1]) == MAX_MESSAGE_CHARS - len(
        "token=<REDACTED_TOKEN>"
    )


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
