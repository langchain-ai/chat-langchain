"""Tests for ingress guards input caps and root-trace metadata helpers."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.message import add_messages

os.environ["USE_LOCAL_PROMPTS"] = "1"

from src.middleware.ingress_guards_middleware import (
    MAX_MESSAGE_CHARS,
    IngressGuardsMiddleware,
)
from src.utils.secret_redaction import REDACTED_SECRET
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


@pytest.mark.parametrize(
    "content",
    [
        "Hello",
        'aws_secret_access_key: "your_secret"',
        "aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']",
        "postgres://user:<your password>@host/db",
        "postgresql://user:settings.X@host/db",
    ],
)
def test_before_agent_noop_when_under_cap(content):
    middleware = IngressGuardsMiddleware()
    state = {"messages": [HumanMessage(content=content, id="h1")]}

    assert middleware.before_agent(state, runtime=SimpleNamespace()) is None


@pytest.mark.parametrize(
    "content, expected",
    [
        ("lsv2_pt_" + "a" * 32, REDACTED_SECRET),
        (
            "aws_secret_access_key=synthetic-secret",
            f"aws_secret_access_key={REDACTED_SECRET}",
        ),
        (
            'aws_secret_access_key: "synthetic-secret"',
            f'aws_secret_access_key: "{REDACTED_SECRET}"',
        ),
        (
            "postgres://user:synthetic-password@host/db",
            f"postgres://user:{REDACTED_SECRET}@host/db",
        ),
        (
            "postgresql://user:synthetic-password@host/db",
            f"postgresql://user:{REDACTED_SECRET}@host/db",
        ),
    ],
)
def test_before_agent_redacts_and_overwrites_message_with_existing_id(
    content, expected
):
    human = HumanMessage(content=content, id="h1")
    messages = [AIMessage(content="hi", id="a1"), human]
    update = IngressGuardsMiddleware().before_agent(
        {"messages": messages}, runtime=SimpleNamespace()
    )

    assert update is not None
    merged = add_messages(messages, update["messages"])
    assert len(merged) == 2
    assert merged[-1].id == "h1"
    assert merged[-1].content == expected


@pytest.mark.parametrize(
    "secret, expected",
    [
        ("sk-proj-" + "a" * 32, REDACTED_SECRET),
        (
            'aws_secret_access_key: "synthetic-secret"',
            f'aws_secret_access_key: "{REDACTED_SECRET}"',
        ),
        (
            "postgres://user:synthetic-password@host/db",
            f"postgres://user:{REDACTED_SECRET}@host/db",
        ),
    ],
)
def test_before_agent_redacts_multimodal_text_and_preserves_media(secret, expected):
    media = {"type": "image_url", "image_url": {"url": "data:image/png;base64,example"}}
    human = HumanMessage(
        content=[secret, {"type": "text", "text": secret, "extra": "kept"}, media],
        id="h1",
    )
    update = IngressGuardsMiddleware().before_agent(
        {"messages": [human]}, runtime=SimpleNamespace()
    )

    assert update is not None
    assert update["messages"][0].id == "h1"
    assert human.content == [
        expected,
        {"type": "text", "text": expected, "extra": "kept"},
        media,
    ]


def test_before_agent_combines_truncation_and_redaction():
    secret = "lsv2_pt_" + "a" * 32
    for content in (
        secret + " " + "z" * MAX_MESSAGE_CHARS,
        [secret, {"type": "text", "text": "z" * MAX_MESSAGE_CHARS}],
    ):
        human = HumanMessage(content=content, id="h1")
        update = IngressGuardsMiddleware().before_agent(
            {"messages": [human]}, runtime=SimpleNamespace()
        )
        assert update is not None
        assert human.id == "h1"
        assert secret not in str(human.content)
        assert REDACTED_SECRET in str(human.content)
        if isinstance(human.content, str):
            assert len(human.content) <= MAX_MESSAGE_CHARS
        else:
            assert (
                sum(
                    len(block if isinstance(block, str) else block["text"])
                    for block in human.content
                )
                <= MAX_MESSAGE_CHARS
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
