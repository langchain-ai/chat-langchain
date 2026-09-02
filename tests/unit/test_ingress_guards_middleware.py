"""Tests for ingress guards input caps and root-trace metadata helpers."""

from __future__ import annotations

import os
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

os.environ["USE_LOCAL_PROMPTS"] = "1"

from src.middleware.ingress_guards_middleware import (
    MAX_MESSAGE_CHARS,
    REDACTED_CREDENTIAL_PLACEHOLDER,
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
    content = "Hello"
    state = {"messages": [HumanMessage(content=content, id="h1")]}

    assert middleware.before_agent(state, runtime=SimpleNamespace()) is None
    assert state["messages"][0].content is content


def test_redacts_langsmith_key_in_plain_string_content():
    middleware = IngressGuardsMiddleware()
    content = "Use lsv2_sk_abcdefghijklmnop_1234567890 in this request."
    human = HumanMessage(content=content, id="h1")

    update = middleware.before_agent({"messages": [human]}, runtime=SimpleNamespace())

    assert update["messages"][0].content == (
        f"Use {REDACTED_CREDENTIAL_PLACEHOLDER} in this request."
    )


def test_redacts_text_blocks_and_preserves_non_text_blocks():
    middleware = IngressGuardsMiddleware()
    image_block = {"type": "image_url", "image_url": {"url": "fake"}}
    content = [
        {"type": "text", "text": "Key: lsv2_pt_abcdefghijklmnop_1234567890"},
        image_block,
    ]
    human = HumanMessage(content=content, id="h1")
    original_image_block = human.content[1]

    update = middleware.before_agent({"messages": [human]}, runtime=SimpleNamespace())

    assert update["messages"][0].content[0]["text"] == (
        f"Key: {REDACTED_CREDENTIAL_PLACEHOLDER}"
    )
    assert update["messages"][0].content[1] is original_image_block


def test_redacts_and_truncates_in_one_pass():
    middleware = IngressGuardsMiddleware()
    credential = "lsv2_sk_abcdefghijklmnop_1234567890"
    content = f"{credential} " + ("x" * MAX_MESSAGE_CHARS)
    human = HumanMessage(content=content, id="h1")

    update = middleware.before_agent({"messages": [human]}, runtime=SimpleNamespace())

    result = update["messages"][0].content
    assert credential not in result
    assert REDACTED_CREDENTIAL_PLACEHOLDER in result
    assert len(result) == MAX_MESSAGE_CHARS


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
