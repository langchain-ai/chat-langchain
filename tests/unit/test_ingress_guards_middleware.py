"""Tests for ingress guards: input caps + root trace metadata stamping."""

from __future__ import annotations

import os
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

os.environ["USE_LOCAL_PROMPTS"] = "1"

from src.middleware import ingress_guards_middleware as ingress_module
from src.middleware.ingress_guards_middleware import (
    MAX_MESSAGE_CHARS,
    IngressGuardsMiddleware,
    _agent_version_metadata,
    _root_run_tree,
)


class _FakeRunTree:
    def __init__(self, *, parent_run=None, metadata=None):
        self.parent_run = parent_run
        self.metadata = {} if metadata is None else metadata


def test_root_run_tree_walks_to_outermost_parent():
    root = _FakeRunTree()
    child = _FakeRunTree(parent_run=root)
    leaf = _FakeRunTree(parent_run=child)
    assert _root_run_tree(leaf) is root
    assert _root_run_tree(root) is root


def test_agent_version_prefers_langchain_revision_id(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_REVISION_ID", "rev-a")
    monkeypatch.setenv("LANGSMITH_HOST_REVISION_ID", "rev-b")
    assert _agent_version_metadata() == {"LANGSMITH_AGENT_VERSION": "rev-a"}


def test_agent_version_falls_back_to_host_revision_id(monkeypatch):
    monkeypatch.delenv("LANGCHAIN_REVISION_ID", raising=False)
    monkeypatch.setenv("LANGSMITH_HOST_REVISION_ID", "host-rev")
    assert _agent_version_metadata() == {"LANGSMITH_AGENT_VERSION": "host-rev"}


def test_agent_version_empty_when_unset(monkeypatch):
    monkeypatch.delenv("LANGCHAIN_REVISION_ID", raising=False)
    monkeypatch.delenv("LANGSMITH_HOST_REVISION_ID", raising=False)
    assert _agent_version_metadata() == {}


def test_enrich_metadata_stamps_root_not_child(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_REVISION_ID", "deploy-sha")
    root = _FakeRunTree()
    child = _FakeRunTree(parent_run=root)
    monkeypatch.setattr(ingress_module.ls, "get_current_run_tree", lambda: child)

    middleware = IngressGuardsMiddleware()
    middleware._enrich_metadata()

    assert root.metadata["source_type"] == "Chat-LangChain"
    assert root.metadata["prompt_source"] == "local:src/prompts/docs_agent_prompt.py"
    assert (
        root.metadata["guardrails_prompt_source"]
        == "local:src/prompts/guardrails_prompts.py"
    )
    assert root.metadata["LANGSMITH_AGENT_VERSION"] == "deploy-sha"
    assert child.metadata == {}


def test_enrich_metadata_does_not_overwrite_existing_root_keys(monkeypatch):
    root = _FakeRunTree(
        metadata={
            "source_type": "Studio",
            "prompt_source": "already-set",
            "LANGSMITH_AGENT_VERSION": "keep-me",
        }
    )
    monkeypatch.setattr(ingress_module.ls, "get_current_run_tree", lambda: root)
    monkeypatch.setenv("LANGCHAIN_REVISION_ID", "other")

    IngressGuardsMiddleware()._enrich_metadata()

    assert root.metadata["source_type"] == "Studio"
    assert root.metadata["prompt_source"] == "already-set"
    assert root.metadata["LANGSMITH_AGENT_VERSION"] == "keep-me"


def test_before_agent_truncates_oversized_human_message(monkeypatch):
    monkeypatch.setattr(ingress_module.ls, "get_current_run_tree", lambda: None)
    middleware = IngressGuardsMiddleware()
    long_text = "x" * (MAX_MESSAGE_CHARS + 50)
    human = HumanMessage(content=long_text, id="h1")
    state = {"messages": [AIMessage(content="hi"), human]}

    update = middleware.before_agent(state, runtime=SimpleNamespace())

    assert update is not None
    assert update["messages"][0].id == "h1"
    assert len(update["messages"][0].content) == MAX_MESSAGE_CHARS


def test_before_agent_noop_when_under_cap(monkeypatch):
    monkeypatch.setattr(ingress_module.ls, "get_current_run_tree", lambda: None)
    middleware = IngressGuardsMiddleware()
    state = {"messages": [HumanMessage(content="Hello", id="h1")]}

    assert middleware.before_agent(state, runtime=SimpleNamespace()) is None
