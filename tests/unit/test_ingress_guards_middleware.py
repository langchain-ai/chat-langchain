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


def test_before_agent_redacts_langsmith_key_in_curl_command():
    middleware = IngressGuardsMiddleware()
    content = 'curl -H "X-Api-Key: lsv2_sk_live-secret_123" https://example.com'
    human = HumanMessage(content=content, id="h1")

    update = middleware.before_agent({"messages": [human]}, runtime=SimpleNamespace())

    assert update["messages"][0].content == (
        'curl -H "X-Api-Key: YOUR_API_KEY_HERE" https://example.com'
    )


def test_before_agent_redacts_x_api_key_header_value():
    middleware = IngressGuardsMiddleware()
    human = HumanMessage(content="X-Api-Key: abc-secret-value", id="h1")

    update = middleware.before_agent({"messages": [human]}, runtime=SimpleNamespace())

    assert update["messages"][0].content == "X-Api-Key: YOUR_API_KEY_HERE"


def test_before_agent_preserves_documentation_placeholder():
    middleware = IngressGuardsMiddleware()
    human = HumanMessage(content="Use lsv2_your_api_key_here in this example", id="h1")

    assert (
        middleware.before_agent({"messages": [human]}, runtime=SimpleNamespace())
        is None
    )


def test_before_agent_preserves_surrounding_json_and_code():
    middleware = IngressGuardsMiddleware()
    content = '{"api_key":"lsv2_sk_secret","enabled":true}\nprint("done")'
    human = HumanMessage(content=content, id="h1")

    update = middleware.before_agent({"messages": [human]}, runtime=SimpleNamespace())

    assert update["messages"][0].content == (
        '{"api_key":"YOUR_API_KEY_HERE","enabled":true}\nprint("done")'
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
