"""Tests for documentation relevance checks in tool retry middleware."""

import asyncio
import json

from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from src.middleware.tool_retry_middleware import ToolRetryMiddleware


def _request(tool_name: str, args: dict[str, str]) -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"name": tool_name, "id": "call-1", "args": args},
        tool=None,
        state=None,
        runtime=None,
    )


def _run(request: ToolCallRequest, message: ToolMessage) -> ToolMessage:
    async def handler(_request):
        return message

    return asyncio.run(ToolRetryMiddleware().awrap_tool_call(request, handler))


def test_replaces_zero_relevance_docs_search_result():
    result = _run(
        _request("search_docs_by_lang_chain", {"query": "widget integration"}),
        ToolMessage(
            content="Unrelated deployment documentation", tool_call_id="call-1"
        ),
    )

    assert json.loads(result.content) == {
        "error": "No matching documentation",
        "tool": "search_docs_by_lang_chain",
        "query": "widget integration",
        "message": "The documentation search returned no page matching this query.",
        "suggestion": (
            "Broaden or rephrase the query, or tell the user no documentation was "
            "found for this topic."
        ),
    }


def test_passes_relevant_docs_search_result_through_byte_for_byte():
    content = "Page: Widget integration\nDetails: configure the widget."
    result = _run(
        _request("search_docs_by_lang_chain", {"query": "widget integration"}),
        ToolMessage(content=content, tool_call_id="call-1"),
    )

    assert result.content == content


def test_does_not_rewrite_non_docs_tool_results():
    content = "Unrelated deployment documentation"
    for tool_name, args in (
        ("check_links", {"urls": "widget integration"}),
        ("search_support_articles", {"query": "widget integration"}),
    ):
        result = _run(
            _request(tool_name, args),
            ToolMessage(content=content, tool_call_id="call-1"),
        )

        assert result.content == content
