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
    password = "vR7!qP2#nL9_xK4%zM8"
    content = (
        "postgresql://app:" + password + "@db.example.test:5432/app?sslmode=require"
    )

    redacted = middleware._redact_secrets(content)

    assert redacted == (
        "postgresql://app:[REDACTED]@db.example.test:5432/app?sslmode=require"
    )


def test_leaves_documentation_placeholders_unchanged():
    middleware = IngressGuardsMiddleware()
    content = (
        "postgresql://postgres:postgres@localhost:5432/db ${DB_PASSWORD} <password>"
    )

    assert middleware._redact_secrets(content) is content


def test_redacts_bearer_token():
    middleware = IngressGuardsMiddleware()
    content = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"

    assert middleware._redact_secrets(content) == "Authorization: Bearer [REDACTED]"


def test_before_agent_redacts_and_truncates_same_message():
    middleware = IngressGuardsMiddleware()
    password = "aB3$dE5!fG7_hI9#jK2"
    content = (
        "postgresql://app:"
        + password
        + "@db.example.test:5432/db"
        + ("x" * MAX_MESSAGE_CHARS)
    )
    human = HumanMessage(content=content, id="h1")

    update = middleware.before_agent({"messages": [human]}, runtime=SimpleNamespace())

    assert update is not None
    assert update["messages"][0].content.startswith(
        "postgresql://app:[REDACTED]@db.example.test:5432/db"
    )
    assert len(update["messages"][0].content) == MAX_MESSAGE_CHARS


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
