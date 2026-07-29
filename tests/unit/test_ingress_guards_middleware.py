"""Tests for ingress guards input caps and root-trace metadata helpers."""

from __future__ import annotations

import os
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

os.environ["USE_LOCAL_PROMPTS"] = "1"

from src.middleware.ingress_guards_middleware import (
    MAX_MESSAGE_CHARS,
    REDACTION_TOKEN,
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


def test_before_agent_redacts_credential_in_string_content():
    middleware = IngressGuardsMiddleware()
    secret = "sk-abcdef0123456789ABCDEF"
    human = HumanMessage(content=f'client = OpenAI(api_key="{secret}")', id="h1")
    state = {"messages": [human]}

    update = middleware.before_agent(state, runtime=SimpleNamespace())

    assert update is not None
    content = update["messages"][0].content
    assert secret not in content
    assert REDACTION_TOKEN in content


def test_before_agent_redacts_text_blocks_and_leaves_others_untouched():
    middleware = IngressGuardsMiddleware()
    secret = "ghp_abcdefghijklmnopqrstuvwxyz0123"
    image_block = {"type": "image_url", "image_url": {"url": "https://x/y.png"}}
    human = HumanMessage(
        content=[{"type": "text", "text": f"token={secret}"}, image_block],
        id="h1",
    )
    state = {"messages": [human]}

    update = middleware.before_agent(state, runtime=SimpleNamespace())

    assert update is not None
    blocks = update["messages"][0].content
    assert blocks[0]["text"] == f"token={REDACTION_TOKEN}"
    assert blocks[1] == image_block


def test_before_agent_leaves_placeholder_keys_alone():
    middleware = IngressGuardsMiddleware()
    state = {
        "messages": [
            HumanMessage(
                content='api_key="sk-YOUR_API_KEY_HERE" and "sk-xxxxxxxxxxxxxxxxxxxx"',
                id="h1",
            )
        ]
    }

    assert middleware.before_agent(state, runtime=SimpleNamespace()) is None


def test_before_agent_redacts_before_truncating_oversized_message():
    middleware = IngressGuardsMiddleware()
    secret = "sk-abcdef0123456789ABCDEF"
    human = HumanMessage(
        content=f"key={secret}\n" + "x" * (MAX_MESSAGE_CHARS + 50),
        id="h1",
    )
    state = {"messages": [human]}

    update = middleware.before_agent(state, runtime=SimpleNamespace())

    assert update is not None
    content = update["messages"][0].content
    assert len(content) == MAX_MESSAGE_CHARS
    assert secret not in content
    assert REDACTION_TOKEN in content


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
