"""Tests for the citation-grounding guard on the docs agent's final answer."""

from __future__ import annotations

import os
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

os.environ["USE_LOCAL_PROMPTS"] = "1"

from src.middleware.link_grounding_middleware import LinkGroundingMiddleware

RETRIEVAL_RESULT = """Results:
Page: LangGraph Persistence
Link: https://docs.langchain.com/oss/python/langgraph/persistence
"""


def _answer(*urls: str) -> AIMessage:
    entries = "\n".join(f"- [Docs]({url})" for url in urls)
    return AIMessage(
        content=f"**Use a checkpointer.**\n\nDetails here.\n\n**Relevant docs:**\n{entries}",
        id="a1",
    )


def test_footer_with_retrieved_urls_is_unchanged():
    middleware = LinkGroundingMiddleware()
    state = {
        "messages": [
            HumanMessage(content="How do I persist state?"),
            ToolMessage(
                content=RETRIEVAL_RESULT,
                name="search_docs_by_lang_chain",
                tool_call_id="t1",
            ),
            _answer("https://docs.langchain.com/oss/python/langgraph/persistence"),
        ]
    }

    assert middleware.after_model(state, runtime=SimpleNamespace()) is None


def test_unretrieved_and_unvalidated_url_is_stripped():
    middleware = LinkGroundingMiddleware()
    state = {
        "messages": [
            HumanMessage(content="How do I persist state?"),
            ToolMessage(
                content=RETRIEVAL_RESULT,
                name="search_docs_by_lang_chain",
                tool_call_id="t1",
            ),
            _answer(
                "https://docs.langchain.com/oss/python/langgraph/persistence",
                "https://docs.langchain.com/oss/python/langgraph/made-up-page",
            ),
        ]
    }

    update = middleware.after_model(state, runtime=SimpleNamespace())

    assert update is not None
    content = update["messages"][0].content
    assert "made-up-page" not in content
    assert "langgraph/persistence" in content
    assert "**Relevant docs:**" in content


def test_check_links_validated_url_is_kept():
    middleware = LinkGroundingMiddleware()
    validated = "https://docs.langchain.com/oss/python/langgraph/persistence#ttl"
    state = {
        "messages": [
            HumanMessage(content="How do I persist state?"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "check_links",
                        "args": {"urls": [validated, "https://docs.langchain.com/dead"]},
                        "id": "t1",
                    }
                ],
            ),
            ToolMessage(
                content=(
                    "Link Check Results: 1/2 valid\n\n"
                    "Invalid links:\n  - https://docs.langchain.com/dead: HTTP 404\n\n"
                    f"Valid links:\n  - {validated}"
                ),
                name="check_links",
                tool_call_id="t1",
            ),
            _answer(validated),
        ]
    }

    assert middleware.after_model(state, runtime=SimpleNamespace()) is None


def test_invalid_check_links_url_is_stripped():
    middleware = LinkGroundingMiddleware()
    state = {
        "messages": [
            HumanMessage(content="How do I persist state?"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "check_links",
                        "args": {"urls": ["https://docs.langchain.com/dead"]},
                        "id": "t1",
                    }
                ],
            ),
            ToolMessage(
                content=(
                    "Link Check Results: 0/1 valid\n\n"
                    "Invalid links:\n  - https://docs.langchain.com/dead: HTTP 404"
                ),
                name="check_links",
                tool_call_id="t1",
            ),
            _answer("https://docs.langchain.com/dead"),
        ]
    }

    update = middleware.after_model(state, runtime=SimpleNamespace())

    assert update is not None
    content = update["messages"][0].content
    assert "docs.langchain.com/dead" not in content
    assert "Relevant docs" not in content


def test_message_without_relevant_docs_marker_is_untouched():
    middleware = LinkGroundingMiddleware()
    state = {
        "messages": [
            HumanMessage(content="hi"),
            AIMessage(content="**Hello!** See https://docs.langchain.com/made-up", id="a1"),
        ]
    }

    assert middleware.after_model(state, runtime=SimpleNamespace()) is None


def test_log_only_mode_leaves_content_intact():
    middleware = LinkGroundingMiddleware(strip_unverified_links=False)
    state = {
        "messages": [
            HumanMessage(content="How do I persist state?"),
            _answer("https://docs.langchain.com/oss/python/langgraph/made-up-page"),
        ]
    }

    assert middleware.after_model(state, runtime=SimpleNamespace()) is None
