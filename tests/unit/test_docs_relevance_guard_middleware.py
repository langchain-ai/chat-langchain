"""Tests for documentation search relevance miss markers."""

import asyncio

from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from src.middleware.docs_relevance_guard_middleware import (
    DocsRelevanceGuardMiddleware,
)


def _request(name: str, args: dict, call_id: str = "call-1"):
    return ToolCallRequest(
        tool_call={"name": name, "id": call_id, "args": args},
        tool=None,
        state=None,
        runtime=None,
    )


def _invoke(request, content: str):
    async def handler(call):
        return ToolMessage(
            content=content,
            name=call.tool_call["name"],
            tool_call_id=call.tool_call["id"],
        )

    return asyncio.run(DocsRelevanceGuardMiddleware().awrap_tool_call(request, handler))


def test_matching_search_results_are_not_marked():
    result = _invoke(
        _request("search_docs_by_lang_chain", {"query": "middleware"}),
        "Page: Middleware\nContent: Configure middleware here.",
    )

    assert result.content == "Page: Middleware\nContent: Configure middleware here."
    assert "NO MATCHING DOCUMENTATION" not in result.content


def test_missing_search_terms_are_marked_and_original_results_preserved():
    original = "Page: Deployment\nContent: Deploy your application."
    result = _invoke(
        _request("search_docs_by_lang_chain", {"query": "middleware callbacks"}),
        original,
    )

    assert result.content.startswith("NO MATCHING DOCUMENTATION: the docs search")
    assert original in result.content
    assert "middleware callbacks" in result.content


def test_non_documentation_search_is_untouched():
    original = "Support article result"
    result = _invoke(
        _request("search_support_articles", {"query": "middleware"}),
        original,
    )

    assert result.content == original
