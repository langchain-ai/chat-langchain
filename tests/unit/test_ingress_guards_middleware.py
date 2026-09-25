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


def test_before_agent_redacts_api_key_in_plain_string_message():
    middleware = IngressGuardsMiddleware()
    human = HumanMessage(content="Use sk-" + "a" * 51, id="h1")

    update = middleware.before_agent({"messages": [human]}, runtime=SimpleNamespace())

    assert update is not None
    assert update["messages"][0].content == f"Use {REDACTION_PLACEHOLDER}"


def test_before_agent_redacts_text_block_and_preserves_image_block():
    middleware = IngressGuardsMiddleware()
    image_block = {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,abc"},
    }
    human = HumanMessage(
        content=[
            {"type": "text", "text": "Token: sk-" + "b" * 51},
            image_block,
        ],
        id="h1",
    )
    original_image_block = human.content[1]

    update = middleware.before_agent({"messages": [human]}, runtime=SimpleNamespace())

    assert update is not None
    assert update["messages"][0].content[0]["text"] == f"Token: {REDACTION_PLACEHOLDER}"
    assert update["messages"][0].content[1] == original_image_block
    assert update["messages"][0].content[1] is original_image_block


def test_redact_secrets_returns_identical_object_without_secret():
    middleware = IngressGuardsMiddleware()
    content = ["No credentials here", {"type": "image_url", "url": "image"}]

    assert middleware._redact_secrets(content) is content


def test_before_agent_redacts_secret_before_truncation_boundary():
    middleware = IngressGuardsMiddleware()
    prefix = "x" * (MAX_MESSAGE_CHARS - len(REDACTION_PLACEHOLDER))
    human = HumanMessage(content=prefix + "sk-" + "c" * 51 + "tail", id="h1")

    update = middleware.before_agent({"messages": [human]}, runtime=SimpleNamespace())

    assert update is not None
    assert REDACTION_PLACEHOLDER in update["messages"][0].content


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
