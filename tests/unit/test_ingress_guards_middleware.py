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


def test_before_agent_redacts_credentials_in_fenced_code():
    middleware = IngressGuardsMiddleware()
    secret = "sk-1234567890abcdefghijkl"
    human = HumanMessage(
        content=f"```python\nclient = OpenAI(api_key='{secret}')\n```", id="h1"
    )

    update = middleware.before_agent({"messages": [human]}, runtime=SimpleNamespace())

    assert update is not None
    assert secret not in update["messages"][0].content
    assert "<REDACTED_API_KEY>" in update["messages"][0].content


def test_before_agent_redacts_text_blocks_and_preserves_non_text_blocks():
    middleware = IngressGuardsMiddleware()
    secret = "ghp_123456789012345678901234567890123456"
    image = {"type": "image_url", "image_url": {"url": "data:image/png"}}
    human = HumanMessage(
        content=[{"type": "text", "text": f"token={secret}"}, image], id="h1"
    )

    update = middleware.before_agent({"messages": [human]}, runtime=SimpleNamespace())

    assert update is not None
    assert secret not in update["messages"][0].content[0]["text"]
    assert update["messages"][0].content[1] == image


def test_before_agent_leaves_credential_free_content_unchanged():
    middleware = IngressGuardsMiddleware()
    human = HumanMessage(content="No credentials here", id="h1")

    assert (
        middleware.before_agent({"messages": [human]}, runtime=SimpleNamespace())
        is None
    )


def test_before_agent_redacts_generic_assignment_value():
    middleware = IngressGuardsMiddleware()
    secret = "super-secret-provider-value"
    human = HumanMessage(content=f"Authorization: Bearer {secret}", id="h1")

    update = middleware.before_agent({"messages": [human]}, runtime=SimpleNamespace())

    assert update is not None
    assert secret not in update["messages"][0].content
    assert update["messages"][0].content == "Authorization: Bearer <REDACTED_API_KEY>"


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
