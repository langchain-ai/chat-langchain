"""Tests for ingress guards input caps and root-trace metadata helpers."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
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
    capped = update["messages"][0].content
    assert len(capped) == MAX_MESSAGE_CHARS
    assert "[Input truncated:" in capped
    assert str(len(long_text)) in capped
    assert "the rest was not received" in capped
    assert "smaller parts" in capped
    assert middleware.before_agent(state, runtime=SimpleNamespace()) is None


@pytest.mark.parametrize("length", [0, 20, MAX_MESSAGE_CHARS])
@pytest.mark.parametrize("as_blocks", [False, True])
def test_before_agent_noop_when_within_cap(length, as_blocks):
    middleware = IngressGuardsMiddleware()
    content = "x" * length
    if as_blocks:
        content = [
            content[:10],
            {"type": "text", "text": content[10:]},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.png"},
            },
        ]
    human = HumanMessage(content=content, id="h1")
    original_content = human.content
    state = {"messages": [human]}

    assert middleware.before_agent(state, runtime=SimpleNamespace()) is None
    assert human.content is original_content
    assert human.content == content


@pytest.mark.parametrize("cut_string_block", [False, True])
@pytest.mark.parametrize("prefix_length", [10, MAX_MESSAGE_CHARS - 10])
def test_before_agent_truncates_text_blocks_and_preserves_non_text(
    cut_string_block, prefix_length
):
    middleware = IngressGuardsMiddleware()
    image = {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}
    long_text = "y" * MAX_MESSAGE_CHARS
    content = [
        "x" * prefix_length,
        image,
        long_text
        if cut_string_block
        else {"type": "text", "text": long_text, "id": "t1"},
        image,
        {"type": "text", "text": "tail"},
        image,
    ]
    human = HumanMessage(content=content, id="h1")

    update = middleware.before_agent({"messages": [human]}, runtime=SimpleNamespace())

    assert update is not None
    assert update["messages"][0].id == "h1"
    capped = update["messages"][0].content
    text_blocks = [
        b if isinstance(b, str) else b["text"]
        for b in capped
        if isinstance(b, str) or b["type"] == "text"
    ]
    text = "".join(text_blocks)
    assert len(text) == MAX_MESSAGE_CHARS
    assert text.count("[Input truncated:") == 1
    cut_index = 0 if prefix_length == MAX_MESSAGE_CHARS - 10 else 1
    assert "[Input truncated:" in text_blocks[cut_index]
    assert str(prefix_length + len(long_text) + len("tail")) in text
    assert "the rest was not received" in text
    assert "smaller parts" in text
    assert text_blocks[-1] == ""
    assert [capped[i] for i in (1, 3, 5)] == [image, image, image]
    if not cut_string_block:
        assert capped[2]["id"] == "t1"
    assert content[2] == (
        long_text
        if cut_string_block
        else {"type": "text", "text": long_text, "id": "t1"}
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
