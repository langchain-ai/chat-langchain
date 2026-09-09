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
from src.utils.secret_redaction import redact_secrets
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


def test_redact_secrets_preserves_prefix_in_curl_header():
    token = "lsv2_sk_live_1234567890abcdef"
    text, count = redact_secrets(f'curl -H "X-Api-Key: {token}" https://example.com')

    assert text == 'curl -H "X-Api-Key: lsv2_sk_<REDACTED>" https://example.com'
    assert count == 1
    assert token not in text


def test_redact_secrets_leaves_placeholders_unchanged():
    text = "api_key=YOUR_API_KEY_HERE"

    assert redact_secrets(text) == (text, 0)


def test_redact_secrets_leaves_normal_text_unchanged():
    text = 'Use `curl -H \\"Accept: application/json\\"` for the request.'

    assert redact_secrets(text) == (text, 0)


def test_before_agent_redacts_text_blocks_and_historical_messages():
    old_token = "sk-old_1234567890abcdef"
    new_token = "tvly-new_1234567890abcdef"
    old = HumanMessage(content=f"old token: {old_token}", id="old")
    latest = HumanMessage(
        content=[
            {"type": "text", "text": f"new token: {new_token}"},
            {"type": "image", "url": "image.png"},
        ],
        id="latest",
    )

    update = IngressGuardsMiddleware().before_agent(
        {"messages": [old, latest]}, runtime=SimpleNamespace()
    )

    assert update is not None
    assert {message.id for message in update["messages"]} == {"old", "latest"}
    assert old_token not in old.content
    assert new_token not in latest.content[0]["text"]
    assert latest.content[1] == {"type": "image", "url": "image.png"}


def test_before_agent_returns_none_without_redaction_or_truncation():
    text = "No credentials here."
    state = {"messages": [HumanMessage(content=text, id="h1")]}

    assert (
        IngressGuardsMiddleware().before_agent(state, runtime=SimpleNamespace()) is None
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
