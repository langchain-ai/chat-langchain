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


def test_before_agent_redacts_lsv2_secret_in_curl_snippet():
    middleware = IngressGuardsMiddleware()
    secret = "lsv2_sk_" + "a" * 32
    human = HumanMessage(
        content=f'curl -H "X-Api-Key: {secret}" https://api.smith.langchain.com',
        id="h1",
    )

    update = middleware.before_agent({"messages": [human]}, runtime=SimpleNamespace())

    assert update is not None
    assert secret not in update["messages"][0].content
    assert "<REDACTED_API_KEY>" in update["messages"][0].content


def test_redaction_leaves_ordinary_prose_and_code_unchanged():
    middleware = IngressGuardsMiddleware()
    content = "Use `api_key` as the variable name, not a real credential."

    assert middleware._redact_secrets(content) == content


def test_redaction_does_not_mangle_placeholder():
    middleware = IngressGuardsMiddleware()

    assert middleware._redact_secrets("api_key=YOUR_API_KEY") == "api_key=YOUR_API_KEY"


def test_redaction_handles_list_content_blocks():
    middleware = IngressGuardsMiddleware()
    secret = "ghp_" + "b" * 32
    content = [
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
        {"type": "text", "text": f"token: {secret}"},
        "No credential here.",
    ]

    redacted = middleware._redact_secrets(content)

    assert redacted[0] is content[0]
    assert redacted[1]["text"] == "<REDACTED_API_KEY>"
    assert redacted[2] == content[2]


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
