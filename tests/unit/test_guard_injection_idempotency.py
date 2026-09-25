"""Regression tests for satisfied-aware documentation guards."""

import asyncio

from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.middleware.citation_guard_middleware import CitationGuardMiddleware
from src.middleware.docs_research_guard_middleware import DocsResearchGuardMiddleware


def test_grounded_footer_is_returned_without_an_extra_model_call(monkeypatch):
    url = "https://docs.langchain.com/oss/python/langgraph/graph-api"
    middleware = CitationGuardMiddleware()
    calls = []

    async def handler(request):
        calls.append(request)
        return ModelResponse(
            result=[AIMessage(content=f"Answer\n\nRelevant docs:\n- [Guide]({url})")]
        )

    async def check_urls(urls, timeout):
        raise AssertionError("grounded URLs should not be checked again")

    monkeypatch.setattr(
        "src.middleware.citation_guard_middleware._check_urls_async", check_urls
    )
    request = ModelRequest(
        model=object(),
        messages=[
            HumanMessage(content="Explain StateGraph."),
            ToolMessage(
                content=f"Found {url}",
                name="search_docs_by_lang_chain",
                tool_call_id="search",
            ),
        ],
    )

    result = asyncio.run(middleware.awrap_model_call(request, handler))

    assert len(calls) == 1
    assert result.result[0].content.endswith(f"[Guide]({url})")


def test_docs_search_on_current_turn_skips_research_injection():
    middleware = DocsResearchGuardMiddleware()
    calls = []
    request = ModelRequest(
        model=object(),
        messages=[
            HumanMessage(content="Explain the StateGraph constructor."),
            ToolMessage(
                content="StateGraph accepts nodes and edges.",
                name="search_docs_by_lang_chain",
                tool_call_id="search",
            ),
        ],
    )

    async def handler(request):
        calls.append(request)
        return ModelResponse(
            result=[
                AIMessage(
                    content="The StateGraph constructor accepts configuration options."
                )
            ]
        )

    asyncio.run(middleware.awrap_model_call(request, handler))

    assert len(calls) == 1
    assert all(
        "research this question on this turn" not in message.content
        for message in calls[0].messages
        if isinstance(message, HumanMessage)
    )


def test_docs_research_guard_injects_at_most_once():
    middleware = DocsResearchGuardMiddleware()
    calls = []
    request = ModelRequest(
        model=object(),
        messages=[HumanMessage(content="Explain the StateGraph constructor.")],
    )

    async def handler(request):
        calls.append(request)
        return ModelResponse(
            result=[
                AIMessage(
                    content="The StateGraph constructor accepts configuration options."
                )
            ]
        )

    asyncio.run(middleware.awrap_model_call(request, handler))

    assert len(calls) == 2


def test_citation_guard_injects_at_most_once():
    middleware = CitationGuardMiddleware()
    calls = []
    request = ModelRequest(
        model=object(),
        messages=[HumanMessage(content="Explain the StateGraph constructor.")],
    )

    async def handler(request):
        calls.append(request)
        return ModelResponse(
            result=[
                AIMessage(
                    content=(
                        "Answer\n\nRelevant docs:\n"
                        "- [Guide](https://docs.langchain.com/unverified)"
                    )
                )
            ]
        )

    asyncio.run(middleware.awrap_model_call(request, handler))

    assert len(calls) == 2
