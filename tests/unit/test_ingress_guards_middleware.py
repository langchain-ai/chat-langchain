"""Tests for ingress guards input caps and root-trace metadata helpers."""

from __future__ import annotations

import os
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

os.environ["USE_LOCAL_PROMPTS"] = "1"

from src.middleware.ingress_guards_middleware import (
    MAX_MESSAGE_CHARS,
    REDACTION_PLACEHOLDER,
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


def test_before_agent_redacts_credentials_in_curl_header():
    middleware = IngressGuardsMiddleware()
    token = "lsv2_sk_1234567890abcdef"
    content = f"```bash\ncurl -H 'X-Api-Key: {token}' https://example.com\n```"
    human = HumanMessage(content=content, id="h1")

    update = middleware.before_agent({"messages": [human]}, runtime=SimpleNamespace())

    assert update is not None
    redacted = update["messages"][0].content
    assert REDACTION_PLACEHOLDER in redacted
    assert token not in redacted
    assert "X-Api-Key:" in redacted
    assert "curl -H" in redacted


def test_before_agent_noop_for_code_without_credentials():
    middleware = IngressGuardsMiddleware()
    human = HumanMessage(content="```bash\necho hello\n```", id="h1")

    assert (
        middleware.before_agent({"messages": [human]}, runtime=SimpleNamespace())
        is None
    )


def test_redact_secrets_handles_content_blocks():
    middleware = IngressGuardsMiddleware()
    token = "ghp_12345678901234567890"
    non_text = {"type": "image_url", "image_url": {"url": "image"}}
    content = [
        {"type": "text", "text": f"token={token}"},
        non_text,
        "Authorization: Bearer sk-12345678901234567890",
    ]
    human = HumanMessage(content=content, id="h1")

    update = middleware.before_agent({"messages": [human]}, runtime=SimpleNamespace())

    assert update is not None
    redacted = update["messages"][0].content
    assert redacted[0]["text"] == f"token={REDACTION_PLACEHOLDER}"
    assert redacted[1] == non_text
    assert redacted[2] == f"Authorization: Bearer {REDACTION_PLACEHOLDER}"


def test_before_agent_redacts_before_truncating():
    middleware = IngressGuardsMiddleware()
    token = "lsv2_sk_1234567890abcdef"
    content = f"{token} " + ("x" * MAX_MESSAGE_CHARS)
    human = HumanMessage(content=content, id="h1")

    update = middleware.before_agent({"messages": [human]}, runtime=SimpleNamespace())

    assert update is not None
    redacted = update["messages"][0].content
    assert len(redacted) == MAX_MESSAGE_CHARS
    assert REDACTION_PLACEHOLDER in redacted
    assert token not in redacted


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
