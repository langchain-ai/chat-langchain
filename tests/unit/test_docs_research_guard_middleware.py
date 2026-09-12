"""Tests for fresh documentation research on each technical turn."""

import asyncio

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.middleware.docs_research_guard_middleware import DocsResearchGuardMiddleware


def test_follow_up_turn_forces_research_instead_of_reusing_prior_results():
    middleware = DocsResearchGuardMiddleware()
    calls: list[ModelRequest] = []

    messages = [
        HumanMessage(content="How do I build a graph?"),
        AIMessage(content="Use StateGraph after reading the docs."),
        ToolMessage(
            content="Prior documentation result",
            name="query_docs_filesystem_docs_by_lang_chain",
            tool_call_id="prior-read",
        ),
        HumanMessage(content="What config key does StateGraph accept?"),
    ]

    async def handler(request: ModelRequest) -> ModelResponse:
        calls.append(request)
        if len(calls) == 1:
            return ModelResponse(
                result=[
                    AIMessage(content="StateGraph accepts the configSchema option.")
                ]
            )
        return ModelResponse(
            result=[
                AIMessage(
                    content="I need to verify that in the documentation.",
                    tool_calls=[
                        {
                            "name": "search_docs_by_lang_chain",
                            "args": {"query": "stategraph"},
                            "id": "fresh-search",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        )

    request = ModelRequest(model=object(), messages=messages)
    response = asyncio.run(middleware.awrap_model_call(request, handler))

    assert len(calls) == 2
    assert "research this question on this turn" in calls[1].system_prompt
    assert (
        calls[1].messages[-1].content == "StateGraph accepts the configSchema option."
    )
    assert response.result[0].tool_calls[0]["name"] == "search_docs_by_lang_chain"


def test_check_links_does_not_satisfy_research_requirement():
    middleware = DocsResearchGuardMiddleware()
    calls: list[ModelRequest] = []

    messages = [
        HumanMessage(content="Explain the StateGraph constructor."),
        ToolMessage(content="Link is valid", name="check_links", tool_call_id="link"),
    ]

    async def handler(request: ModelRequest) -> ModelResponse:
        calls.append(request)
        return ModelResponse(
            result=[
                AIMessage(
                    content="The StateGraph constructor accepts configuration options."
                )
            ]
        )

    request = ModelRequest(model=object(), messages=messages)
    asyncio.run(middleware.awrap_model_call(request, handler))

    assert len(calls) == 2


def test_retrieved_and_valid_footer_url_passes_through(monkeypatch):
    from src.middleware import citation_guard_middleware as citation_module
    from src.middleware.citation_guard_middleware import CitationGuardMiddleware

    url = "https://docs.langchain.com/oss/python/langgraph/graph-api"
    middleware = CitationGuardMiddleware()
    response = ModelResponse(
        result=[
            AIMessage(content=f"**Answer**\n\n**Relevant docs:**\n- [Guide]({url})")
        ]
    )
    request = ModelRequest(
        model=object(),
        messages=[
            HumanMessage(content="How do I build a graph?"),
            ToolMessage(
                content=[{"type": "text", "text": f"Retrieved URL: {url}"}],
                name="search_docs_by_lang_chain",
                tool_call_id="search",
            ),
            ToolMessage(
                content=f"Valid links:\n  - {url}",
                name="check_links",
                tool_call_id="check",
            ),
        ],
    )

    async def handler(request: ModelRequest) -> ModelResponse:
        return response

    async def unexpected_check(urls: list[str], timeout: float):
        raise AssertionError("already-valid URL should not be checked again")

    monkeypatch.setattr(citation_module, "_check_urls_async", unexpected_check)
    result = asyncio.run(middleware.awrap_model_call(request, handler))

    assert result.result[0].content.endswith(f"({url})")


def test_ungrounded_footer_url_is_stripped_even_when_reachable(monkeypatch):
    from src.middleware import citation_guard_middleware as citation_module
    from src.middleware.citation_guard_middleware import CitationGuardMiddleware
    from src.tools.link_check_tools import LinkCheckResult

    grounded = "https://docs.langchain.com/oss/python/langgraph/graph-api"
    invented = "https://docs.langchain.com/oss/python/langgraph/invented"
    checks: list[list[str]] = []
    middleware = CitationGuardMiddleware()
    request = ModelRequest(
        model=object(),
        messages=[
            HumanMessage(content="How do I build a graph?"),
            ToolMessage(
                content=f"Retrieved URL: {grounded}",
                name="query_docs_filesystem_docs_by_lang_chain",
                tool_call_id="read",
            ),
        ],
    )

    async def handler(request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            result=[
                AIMessage(
                    content=[
                        {
                            "type": "text",
                            "text": (
                                f"**Answer**\n\n**Relevant docs:**\n"
                                f"- [Guide]({grounded})\n"
                                f"- [Other]({invented})"
                            ),
                        }
                    ]
                )
            ]
        )

    async def check_urls(urls: list[str], timeout: float):
        checks.append(urls)
        return [LinkCheckResult(url=url, valid=True) for url in urls]

    monkeypatch.setattr(citation_module, "_check_urls_async", check_urls)
    result = asyncio.run(middleware.awrap_model_call(request, handler))

    assert checks == [[grounded]]
    assert invented not in result.result[0].content
    assert grounded in result.result[0].content


def test_grounded_unchecked_footer_url_is_validated_before_passing(monkeypatch):
    from src.middleware import citation_guard_middleware as citation_module
    from src.middleware.citation_guard_middleware import CitationGuardMiddleware
    from src.tools.link_check_tools import LinkCheckResult

    url = "https://docs.langchain.com/oss/python/langgraph/graph-api"
    checks: list[list[str]] = []
    middleware = CitationGuardMiddleware()
    request = ModelRequest(
        model=object(),
        messages=[
            HumanMessage(content="How do I build a graph?"),
            ToolMessage(
                content=f"Retrieved URL: {url}",
                name="search_docs_by_lang_chain",
                tool_call_id="search",
            ),
        ],
    )

    async def handler(request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            result=[
                AIMessage(content=f"**Answer**\n\n**Relevant docs:**\n- [Guide]({url})")
            ]
        )

    async def check_urls(urls: list[str], timeout: float):
        checks.append(urls)
        return [LinkCheckResult(url=url, valid=True) for url in urls]

    monkeypatch.setattr(citation_module, "_check_urls_async", check_urls)
    result = asyncio.run(middleware.awrap_model_call(request, handler))

    assert checks == [[url]]
    assert url in result.result[0].content


def test_entirely_ungrounded_footer_retries_with_correction():
    from src.middleware.citation_guard_middleware import CitationGuardMiddleware

    url = "https://docs.langchain.com/oss/python/langgraph/invented"
    middleware = CitationGuardMiddleware()
    calls: list[ModelRequest] = []
    request = ModelRequest(
        model=object(),
        messages=[HumanMessage(content="How do I build a graph?")],
    )

    async def handler(request: ModelRequest) -> ModelResponse:
        calls.append(request)
        return ModelResponse(
            result=[
                AIMessage(content=f"**Answer**\n\n**Relevant docs:**\n- [Guide]({url})")
            ]
        )

    result = asyncio.run(middleware.awrap_model_call(request, handler))

    assert len(calls) == 2
    assert (
        "copied verbatim from this turn's documentation tool results"
        in calls[1].system_prompt
    )
    assert result.result[0].content == calls[1].messages[-1].content


@pytest.mark.parametrize(
    "heading",
    [
        "## Relevant docs:",
        "## Relevant docs",
        "**Relevant docs:**",
        "### 相关文档",
    ],
)
def test_footer_renderings_validate_and_remove_ungrounded_urls(monkeypatch, heading):
    from src.middleware import citation_guard_middleware as citation_module
    from src.middleware.citation_guard_middleware import CitationGuardMiddleware
    from src.tools.link_check_tools import LinkCheckResult

    grounded = "https://docs.langchain.com/oss/python/langgraph/graph-api"
    invented = "https://docs.langchain.com/oss/python/langgraph/invented"
    checks: list[list[str]] = []
    middleware = CitationGuardMiddleware()
    request = ModelRequest(
        model=object(),
        messages=[
            HumanMessage(content="How do I build a graph?"),
            ToolMessage(
                content=f"Retrieved URL: {grounded}",
                name="search_docs_by_lang_chain",
                tool_call_id="search",
            ),
        ],
    )

    async def handler(request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            result=[
                AIMessage(
                    content=(
                        f"Answer\n\n{heading}\n- [Guide]({grounded})\n"
                        f"- [Other]({invented})"
                    )
                )
            ]
        )

    async def check_urls(urls: list[str], timeout: float):
        checks.append(urls)
        return [LinkCheckResult(url=url, valid=True) for url in urls]

    monkeypatch.setattr(citation_module, "_check_urls_async", check_urls)
    result = asyncio.run(middleware.awrap_model_call(request, handler))

    assert checks == [[grounded]]
    assert invented not in result.result[0].content
    assert grounded in result.result[0].content


def test_docs_urls_without_footer_are_validated_and_repaired(monkeypatch):
    from src.middleware import citation_guard_middleware as citation_module
    from src.middleware.citation_guard_middleware import CitationGuardMiddleware
    from src.tools.link_check_tools import LinkCheckResult

    grounded = "https://docs.langchain.com/oss/python/langgraph/graph-api"
    invented = "https://docs.langchain.com/oss/python/langgraph/invented"
    checks: list[list[str]] = []
    middleware = CitationGuardMiddleware()
    request = ModelRequest(
        model=object(),
        messages=[
            HumanMessage(content="How do I build a graph?"),
            ToolMessage(
                content=f"Retrieved URL: {grounded}",
                name="search_docs_by_lang_chain",
                tool_call_id="search",
            ),
        ],
    )

    async def handler(request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            result=[
                AIMessage(
                    content=f"Answer\n- [Guide]({grounded})\n- [Other]({invented})"
                )
            ]
        )

    async def check_urls(urls: list[str], timeout: float):
        checks.append(urls)
        return [LinkCheckResult(url=url, valid=True) for url in urls]

    monkeypatch.setattr(citation_module, "_check_urls_async", check_urls)
    result = asyncio.run(middleware.awrap_model_call(request, handler))

    assert checks == [[grounded]]
    assert invented not in result.result[0].content
    assert grounded in result.result[0].content
