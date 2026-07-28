"""Tests for the docs footer citation-provenance postcondition."""

from __future__ import annotations

import os
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

os.environ["USE_LOCAL_PROMPTS"] = "1"

from src.middleware.citation_provenance_middleware import CitationProvenanceMiddleware

RUNTIME = SimpleNamespace()


def _answer(*links: str) -> str:
    body = "\n".join(f"- {link}" for link in links)
    return (
        "**Use `stream_mode` to stream subgraph output.**\n"
        "\n"
        "Set `subgraphs=True` on the parent stream call.\n"
        "\n"
        "**Relevant docs:**\n"
        f"{body}"
    )


def test_citation_verbatim_in_tool_result_is_kept():
    middleware = CitationProvenanceMiddleware()
    answer = _answer(
        "[Streaming](https://docs.langchain.com/oss/python/langgraph/streaming)"
    )
    state = {
        "messages": [
            HumanMessage(content="how do I stream subgraphs?", id="h1"),
            ToolMessage(
                content="https://docs.langchain.com/oss/python/langgraph/streaming",
                tool_call_id="t1",
                id="m1",
            ),
            AIMessage(content=answer, id="a1"),
        ]
    }

    assert middleware.after_agent(state, RUNTIME) is None


def test_memory_constructed_citation_is_removed_and_rest_is_byte_identical():
    middleware = CitationProvenanceMiddleware()
    grounded = "[Streaming](https://docs.langchain.com/oss/python/langgraph/streaming)"
    invented = "[Tracing](https://docs.langchain.com/oss/python/langsmith/tracing)"
    state = {
        "messages": [
            HumanMessage(content="how do I stream subgraphs?", id="h1"),
            ToolMessage(
                content="/oss/python/langgraph/streaming.mdx",
                tool_call_id="t1",
                id="m1",
            ),
            AIMessage(content=_answer(grounded, invented), id="a1"),
        ]
    }

    update = middleware.after_agent(state, RUNTIME)

    assert update is not None
    assert update["messages"][0].id == "a1"
    assert update["messages"][0].content == _answer(grounded)


def test_url_validated_only_via_check_links_is_kept():
    middleware = CitationProvenanceMiddleware()
    url = "https://docs.langchain.com/oss/python/langgraph/streaming#stream-subgraph-outputs"
    state = {
        "messages": [
            HumanMessage(content="how do I stream subgraphs?", id="h1"),
            AIMessage(
                content="",
                id="a0",
                tool_calls=[
                    {"name": "check_links", "args": {"urls": [url]}, "id": "t1"}
                ],
            ),
            ToolMessage(
                content="Link Check Results: 1/1 valid\n\nValid links:\n  - " + url,
                tool_call_id="t1",
                id="m1",
            ),
            AIMessage(content=_answer(f"[Streaming]({url})"), id="a1"),
        ]
    }

    assert middleware.after_agent(state, RUNTIME) is None


def test_content_blocks_are_handled():
    middleware = CitationProvenanceMiddleware()
    invented = "[Tracing](https://docs.langchain.com/oss/python/langsmith/tracing)"
    grounded = "[Streaming](https://docs.langchain.com/oss/python/langgraph/streaming)"
    state = {
        "messages": [
            HumanMessage(content="how do I stream subgraphs?", id="h1"),
            ToolMessage(
                content=[
                    {
                        "type": "text",
                        "text": "https://docs.langchain.com/oss/python/langgraph/streaming",
                    }
                ],
                tool_call_id="t1",
                id="m1",
            ),
            AIMessage(
                content=[{"type": "text", "text": _answer(grounded, invented)}],
                id="a1",
            ),
        ]
    }

    update = middleware.after_agent(state, RUNTIME)

    assert update is not None
    assert update["messages"][0].content == [
        {"type": "text", "text": _answer(grounded)}
    ]


def test_all_citations_ungrounded_drops_the_footer_header():
    middleware = CitationProvenanceMiddleware()
    invented = "[Tracing](https://docs.langchain.com/oss/python/langsmith/tracing)"
    state = {
        "messages": [
            HumanMessage(content="how do I stream subgraphs?", id="h1"),
            AIMessage(content=_answer(invented), id="a1"),
        ]
    }

    update = middleware.after_agent(state, RUNTIME)

    assert update is not None
    content = update["messages"][0].content
    assert "Relevant docs" not in content
    assert content.startswith("**Use `stream_mode` to stream subgraph output.**")


def test_answer_without_footer_is_unchanged():
    middleware = CitationProvenanceMiddleware()
    state = {
        "messages": [
            HumanMessage(content="hi", id="h1"),
            AIMessage(content="**Hello!** How can I help with LangChain?", id="a1"),
        ]
    }

    assert middleware.after_agent(state, RUNTIME) is None
