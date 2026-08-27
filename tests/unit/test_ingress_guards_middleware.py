"""Tests for ingress guards input caps and root-trace metadata helpers."""

from __future__ import annotations

import os
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

os.environ["USE_LOCAL_PROMPTS"] = "1"

from src.middleware.ingress_guards_middleware import (
    MAX_MESSAGE_CHARS,
    IngressGuardsMiddleware,
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


def test_before_agent_redacts_postgres_uri_password_and_preserves_uri():
    middleware = IngressGuardsMiddleware()
    content = (
        "postgres://db-user:p%40ss%3Aword@db.example.com:5432/app"
        "?sslmode=require&connect_timeout=10"
    )
    human = HumanMessage(content=content, id="h1")

    update = middleware.before_agent({"messages": [human]}, runtime=SimpleNamespace())

    assert update is not None
    assert (
        update["messages"][0].content
        == "postgres://db-user:<REDACTED_SECRET>@db.example.com:5432/app"
        "?sslmode=require&connect_timeout=10"
    )


def test_redact_secrets_skips_placeholders_and_getenv_calls():
    middleware = IngressGuardsMiddleware()
    content = 'PASSWORD=your-password API_KEY=YOUR_API_KEY_HERE os.getenv("X")'

    assert middleware._redact_secrets(content) == content


def test_redact_secrets_handles_list_content_blocks():
    middleware = IngressGuardsMiddleware()
    content = [
        {"type": "text", "text": "DATABASE_URL=postgres://user:live-pass@host/db"},
        {"type": "image", "url": "https://example.com/image.png"},
        "token=live-token-value",
    ]

    redacted = middleware._redact_secrets(content)

    assert redacted == [
        {
            "type": "text",
            "text": "DATABASE_URL=postgres://user:<REDACTED_SECRET>@host/db",
        },
        {"type": "image", "url": "https://example.com/image.png"},
        "token=<REDACTED_SECRET>",
    ]


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
