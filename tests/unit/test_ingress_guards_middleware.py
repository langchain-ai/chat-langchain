"""Tests for ingress guards input caps and root-trace metadata helpers."""

from __future__ import annotations

import json
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


def test_before_agent_rejects_misrouted_agent_state():
    middleware = IngressGuardsMiddleware()
    blob = json.dumps(
        {
            "name": "leadership_coach_agent",
            "thread_model_call_count": 3,
            "off_topic_query": False,
            "messages": [
                {"type": "human", "content": "How do I recognize my team?"},
                {
                    "type": "tool",
                    "name": "team_recognition_history",
                    "content": "Prior recognitions: ...",
                },
            ],
        }
    )
    state = {"messages": [HumanMessage(content=blob, id="h1")]}

    update = middleware.before_agent(state, runtime=SimpleNamespace())

    assert update is not None
    assert update["jump_to"] == "end"
    assert isinstance(update["messages"][0], AIMessage)


def test_before_agent_rejects_misrouted_agent_state_under_values_wrapper():
    middleware = IngressGuardsMiddleware()
    blob = json.dumps(
        {
            "values": {
                "name": "leadership_coach_agent",
                "messages": [
                    {
                        "type": "tool",
                        "name": "team_recognition_history",
                        "content": "Prior recognitions: ...",
                    }
                ],
            }
        }
    )
    state = {"messages": [HumanMessage(content=blob, id="h1")]}

    update = middleware.before_agent(state, runtime=SimpleNamespace())

    assert update is not None
    assert update["jump_to"] == "end"


def test_before_agent_allows_question_with_json_snippet():
    middleware = IngressGuardsMiddleware()
    question = (
        "How do I configure a LangChain agent? Here is my config: "
        '{"model": "gpt-4o", "temperature": 0}'
    )
    state = {"messages": [HumanMessage(content=question, id="h1")]}

    assert middleware.before_agent(state, runtime=SimpleNamespace()) is None


def test_build_docs_agent_trace_metadata_includes_provenance_and_version(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_REVISION_ID", "rev-a")
    monkeypatch.setenv("LANGSMITH_HOST_REVISION_ID", "rev-b")
    monkeypatch.setattr(
        "src.utils.prompt_provenance._USE_LOCAL_PROMPTS",
        True,
    )

    metadata = build_docs_agent_trace_metadata()

    assert metadata["source_type"] == "Chat-LangChain"
    assert metadata["prompt_source"] == "local:src/prompts/docs_agent_prompt.py"
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
