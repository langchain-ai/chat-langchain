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


def test_redact_secrets_preserves_postgres_uri_structure():
    password = "N7v!q2R#k9Lm4Zp8"

    redacted, count = redact_secrets(
        f"postgres://db-user:{password}@db.example.test:5432/docs"
    )

    assert count == 1
    assert redacted == (
        "postgres://db-user:[REDACTED_CREDENTIAL]@db.example.test:5432/docs"
    )


def test_redact_secrets_skips_placeholders_and_environment_references():
    text = 'postgres://user:password@localhost:5432/db os.getenv("OPENAI_API_KEY")'

    redacted, count = redact_secrets(text)

    assert count == 0
    assert redacted == text


def test_redact_secrets_redacts_openai_api_key():
    key = "sk-7xQ2mN8pR4tY6wE9uI3oP5aS"

    redacted, count = redact_secrets(f"key={key}")

    assert count == 1
    assert redacted == "key=[REDACTED_CREDENTIAL]"


def test_before_agent_redacts_content_blocks():
    secret = "ghp_AbCdEfGhIjKlMnOpQrStUvW1"
    human = HumanMessage(
        content=[{"type": "text", "text": f"token: {secret}"}, {"type": "image"}],
        id="h1",
    )

    update = IngressGuardsMiddleware().before_agent(
        {"messages": [human]}, runtime=SimpleNamespace()
    )

    assert update is not None
    assert update["messages"][0].id == "h1"
    assert update["messages"][0].content == [
        {"type": "text", "text": "token: [REDACTED_CREDENTIAL]"},
        {"type": "image"},
    ]


def test_before_agent_redacts_before_truncating():
    password = "N7v!q2R#k9Lm4Zp8Hs6Jd3Kf1Wx9Bc5T"
    uri = f"postgres://db-user:{password}@db.example.test"
    human = HumanMessage(content=("x" * (MAX_MESSAGE_CHARS - 65)) + uri, id="h1")

    update = IngressGuardsMiddleware().before_agent(
        {"messages": [human]}, runtime=SimpleNamespace()
    )

    assert update is not None
    content = update["messages"][0].content
    assert len(content) <= MAX_MESSAGE_CHARS
    assert password not in content
    assert "[REDACTED_CREDENTIAL]" in content


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
