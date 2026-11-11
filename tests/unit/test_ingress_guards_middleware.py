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
    assert len(update["messages"][0].content) == MAX_MESSAGE_CHARS


def test_before_agent_noop_when_under_cap():
    middleware = IngressGuardsMiddleware()
    state = {"messages": [HumanMessage(content="Hello", id="h1")]}

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


@pytest.fixture
def run_metadata(monkeypatch):
    metadata = {"existing": True}
    monkeypatch.setattr(
        "src.middleware.ingress_guards_middleware.ls.get_current_run_tree",
        lambda: SimpleNamespace(metadata=metadata),
    )
    return metadata


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "https://example.com?api_key=tvly-test123&limit=2",
            "https://example.com?api_key=[REDACTED_SECRET]&limit=2",
        ),
        (
            'client = OpenAI(api_key="sk-test123")',
            "client = OpenAI(api_key=[REDACTED_SECRET])",
        ),
        *[
            (prefix + "test123", "[REDACTED_SECRET]")
            for prefix in (
                "sk-ant-",
                "tvly-",
                "AIza",
                "ghp_",
                "gho_",
                "xoxb-",
                "pk_live_",
                "lsv2_",
                "lcl_",
            )
        ],
        ("Authorization: Bearer abc.def-123", "Authorization: [REDACTED_SECRET]"),
        ("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.signature", "[REDACTED_SECRET]"),
        *[
            (name + "=opaque123", name + "=[REDACTED_SECRET]")
            for name in (
                "key",
                "token",
                "apikey",
                "api_key",
                "tavilyApiKey",
                "LANGSMITH_API_KEY",
            )
        ],
    ],
)
def test_before_agent_redacts_credentials(text, expected, run_metadata):
    human = HumanMessage(content=text, id="h1")
    update = IngressGuardsMiddleware().before_agent(
        {"messages": [human]}, runtime=SimpleNamespace()
    )

    assert update["messages"][0].content == expected
    assert update["messages"][0].id == "h1"
    assert run_metadata == {"existing": True, "secret_redacted": True}


def test_before_agent_redacts_text_blocks_and_preserves_images(run_metadata):
    image = {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}
    human = HumanMessage(
        content=[
            "tvly-test123",
            {"type": "text", "text": "sk-test123", "extra": "kept"},
            image,
        ],
        id="h1",
    )

    update = IngressGuardsMiddleware().before_agent(
        {"messages": [human]}, runtime=SimpleNamespace()
    )

    assert update["messages"][0].content == [
        "[REDACTED_SECRET]",
        {"type": "text", "text": "[REDACTED_SECRET]", "extra": "kept"},
        image,
    ]
    assert update["messages"][0].id == "h1"
    assert run_metadata["secret_redacted"] is True


@pytest.mark.parametrize(
    "text", ["How do I use LangGraph?", "api_key=[REDACTED_SECRET]"]
)
def test_before_agent_leaves_safe_text_and_metadata_unchanged(text, run_metadata):
    human = HumanMessage(content=text, id="h1")
    assert (
        IngressGuardsMiddleware().before_agent(
            {"messages": [human]}, runtime=SimpleNamespace()
        )
        is None
    )
    assert human.content == text
    assert run_metadata == {"existing": True}


def test_before_agent_caps_before_redacting(run_metadata):
    text = "tvly-test123 " + "x" * MAX_MESSAGE_CHARS + " sk-outsidecap"
    human = HumanMessage(content=text, id="h1")

    update = IngressGuardsMiddleware().before_agent(
        {"messages": [human]}, runtime=SimpleNamespace()
    )

    assert update["messages"][0].content == "[REDACTED_SECRET] " + "x" * (
        MAX_MESSAGE_CHARS - 13
    )
    assert update["messages"][0].id == "h1"
    assert run_metadata["secret_redacted"] is True


def test_redaction_reaches_downstream_middleware_model_and_history(monkeypatch):
    import asyncio

    from langchain.agents import create_agent
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel,
    )
    from langgraph.checkpoint.memory import InMemorySaver

    from src.middleware.guardrails_middleware import GuardrailsMiddleware
    from src.middleware.legal_handoff_middleware import LegalHandoffMiddleware

    seen = []

    def observe(messages):
        text = " ".join(str(message.content) for message in messages)
        assert "tvly-test123" not in text
        assert "[REDACTED_SECRET]" in text

    class LegalObserver(LegalHandoffMiddleware):
        def before_agent(self, state, runtime):
            observe(state["messages"])
            seen.append("legal")
            return super().before_agent(state, runtime)

    class Classifier:
        def with_structured_output(self, schema):
            return self

        async def ainvoke(self, prompt, config=None):
            observe(prompt)
            seen.append("guardrails")
            return {"decision": "ALLOWED", "explanation": "LangChain question"}

    class ModelObserver(BaseCallbackHandler):
        raise_error = True

        def on_chat_model_start(self, serialized, messages, **kwargs):
            observe(messages[0])
            seen.append("model")

    guardrails = GuardrailsMiddleware.__new__(GuardrailsMiddleware)
    guardrails.classifier_llms = [("fake", Classifier())]
    guardrails.block_off_topic = True
    monkeypatch.setattr("src.middleware.guardrails_middleware.ALLOWED_SAMPLE_RATE", 0)
    graph = create_agent(
        FakeMessagesListChatModel(responses=[AIMessage(content="Help")]),
        middleware=[IngressGuardsMiddleware(), LegalObserver(), guardrails],
        checkpointer=InMemorySaver(),
    )
    config = {
        "configurable": {"thread_id": "redaction"},
        "callbacks": [ModelObserver()],
    }
    result = asyncio.run(
        graph.ainvoke(
            {
                "messages": [
                    HumanMessage(content="LangChain search with tvly-test123", id="h1")
                ]
            },
            config,
        )
    )

    assert seen == ["legal", "guardrails", "model"]
    assert result["messages"][0].id == "h1"
    assert result["messages"][0].content == "LangChain search with [REDACTED_SECRET]"
    assert graph.get_state(config).values["messages"][0] == result["messages"][0]
    asyncio.run(
        graph.ainvoke({"messages": [HumanMessage(content="More details?")]}, config)
    )
    assert seen == ["legal", "guardrails", "model"] * 2


def test_list_content_keeps_shared_truncation_budget(run_metadata):
    image = {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}
    human = HumanMessage(
        content=[
            "x" * (MAX_MESSAGE_CHARS - 13),
            {"type": "text", "text": "tvly-test123 extra"},
            image,
            "discarded",
        ],
        id="h1",
    )
    update = IngressGuardsMiddleware().before_agent(
        {"messages": [human]}, runtime=SimpleNamespace()
    )
    assert update["messages"][0].content == [
        "x" * (MAX_MESSAGE_CHARS - 13),
        {"type": "text", "text": "[REDACTED_SECRET] "},
        image,
        "",
    ]
    assert run_metadata["secret_redacted"] is True
