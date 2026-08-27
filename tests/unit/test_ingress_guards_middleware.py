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


def test_redacts_postgres_uri_password_and_preserves_connection_details():
    middleware = IngressGuardsMiddleware()
    content = (
        "postgres://svc_user:p%40ss$word;?@db.example:5432/app?sslmode=disable&x=1"
    )

    redacted = middleware._redact_secrets(content)

    assert redacted == (
        "postgres://svc_user:[REDACTED_SECRET]@db.example:5432/app?sslmode=disable&x=1"
    )


def test_does_not_redact_passwordless_or_placeholder_postgres_uris():
    middleware = IngressGuardsMiddleware()

    assert (
        middleware._redact_secrets("postgres://user@host/db")
        == "postgres://user@host/db"
    )
    assert middleware._redact_secrets("postgres://user:<password>@host/db") == (
        "postgres://user:<password>@host/db"
    )
    assert middleware._redact_secrets("postgres://user:${PGPASSWORD}@host/db") == (
        "postgres://user:${PGPASSWORD}@host/db"
    )


def test_redacts_prefix_shaped_api_key():
    middleware = IngressGuardsMiddleware()

    assert middleware._redact_secrets("api key sk-test-secret-value") == (
        "api key [REDACTED_SECRET]"
    )


def test_redacts_before_truncating_message():
    middleware = IngressGuardsMiddleware()
    content = "x" * (MAX_MESSAGE_CHARS - 100) + " sk-secret-value " + "y" * 150
    human = HumanMessage(content=content, id="h1")

    update = middleware.before_agent({"messages": [human]}, runtime=SimpleNamespace())

    assert update is not None
    assert "sk-secret-value" not in update["messages"][0].content
    assert "[REDACTED_SECRET]" in update["messages"][0].content
    assert len(update["messages"][0].content) == MAX_MESSAGE_CHARS


def test_redacts_text_blocks_and_preserves_non_text_blocks():
    middleware = IngressGuardsMiddleware()
    image = {"type": "image_url", "image_url": {"url": "https://example.com/sk-secret"}}
    content = [
        {"type": "text", "text": "token=secret-value"},
        image,
    ]

    redacted = middleware._redact_secrets(content)

    assert redacted[0] == {"type": "text", "text": "token=[REDACTED_SECRET]"}
    assert redacted[1] is image


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
