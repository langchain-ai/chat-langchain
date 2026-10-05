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


def test_before_agent_redacts_quoted_api_key_header():
    middleware = IngressGuardsMiddleware()
    key = "lsv2_sk_" + "a" * 40
    content = f'''curl https://example.test -H "X-Api-Key: {key}"'''
    human = HumanMessage(content=content, id="h1")

    update = middleware.before_agent({"messages": [human]}, runtime=SimpleNamespace())

    assert update["messages"][0].content == (
        'curl https://example.test -H "X-Api-Key: <REDACTED_CREDENTIAL>"'
    )


def test_redaction_ignores_documentation_placeholders_and_short_keys():
    middleware = IngressGuardsMiddleware()
    content = "lsv2_your_api_key sk-..."

    assert middleware._redact_text(content) == content


def test_redaction_preserves_non_text_content_blocks():
    middleware = IngressGuardsMiddleware()
    key = "lsv2_sk_" + "b" * 40
    content = [
        {"type": "text", "text": f"key: {key}"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
    ]

    redacted = middleware._redact_content(content)

    assert redacted[0]["text"] == "key: <REDACTED_CREDENTIAL>"
    assert redacted[1] is content[1]


def test_before_agent_redacts_before_truncating():
    middleware = IngressGuardsMiddleware()
    key = "lsv2_sk_" + "c" * 40
    content = key + " " + "x" * MAX_MESSAGE_CHARS
    human = HumanMessage(content=content, id="h1")

    update = middleware.before_agent({"messages": [human]}, runtime=SimpleNamespace())

    assert update["messages"][0].content.startswith("<REDACTED_CREDENTIAL>")
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
