import asyncio

from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


def test_successful_pricing_tool_grounds_pricing_footer(monkeypatch):
    from src.middleware import citation_guard_middleware as citation_module
    from src.middleware.citation_guard_middleware import CitationGuardMiddleware
    from src.tools.link_check_tools import LinkCheckResult

    pricing_url = "https://www.langchain.com/pricing"
    calls: list[ModelRequest] = []
    checks: list[list[str]] = []
    middleware = CitationGuardMiddleware()
    request = ModelRequest(
        model=object(),
        messages=[
            HumanMessage(content="What are the pricing options?"),
            ToolMessage(
                content="LangSmith pricing details",
                name="fetch_langchain_pricing",
                tool_call_id="pricing",
            ),
        ],
    )

    async def handler(request: ModelRequest) -> ModelResponse:
        calls.append(request)
        return ModelResponse(
            result=[
                AIMessage(
                    content=f"Answer\n\n**Relevant docs:**\n- [Pricing]({pricing_url})"
                )
            ]
        )

    async def check_urls(urls: list[str], timeout: float):
        checks.append(urls)
        return [LinkCheckResult(url=url, valid=True) for url in urls]

    monkeypatch.setattr(citation_module, "_check_urls_async", check_urls)
    result = asyncio.run(middleware.awrap_model_call(request, handler))

    assert len(calls) == 1
    assert checks == [[pricing_url]]
    assert pricing_url in result.result[0].content


def test_retry_response_strips_ungrounded_footer_url(monkeypatch):
    from src.middleware.citation_guard_middleware import CitationGuardMiddleware

    initial_url = "https://docs.langchain.com/invented/initial"
    retry_url = "https://docs.langchain.com/invented/retry"
    calls: list[ModelRequest] = []
    middleware = CitationGuardMiddleware()
    request = ModelRequest(
        model=object(),
        messages=[HumanMessage(content="Find documentation.")],
    )

    async def handler(request: ModelRequest) -> ModelResponse:
        calls.append(request)
        url = initial_url if len(calls) == 1 else retry_url
        return ModelResponse(
            result=[
                AIMessage(content=f"Answer\n\n**Relevant docs:**\n- [Guide]({url})")
            ]
        )

    result = asyncio.run(middleware.awrap_model_call(request, handler))

    assert len(calls) == 2
    assert retry_url not in result.result[0].content
    assert result.result[0].content == "Answer\n\n**Relevant docs:**"


def test_fragmented_grounded_url_accepts_canonical_footer_url(monkeypatch):
    from src.middleware import citation_guard_middleware as citation_module
    from src.middleware.citation_guard_middleware import CitationGuardMiddleware
    from src.tools.link_check_tools import LinkCheckResult

    footer_url = "https://docs.langchain.com/guide"
    grounded_url = f"{footer_url}#section"
    checked_urls: list[str] = []
    middleware = CitationGuardMiddleware()
    request = ModelRequest(
        model=object(),
        messages=[
            HumanMessage(content="Find documentation."),
            ToolMessage(
                content=f"Retrieved URL: {grounded_url}",
                name="search_docs_by_lang_chain",
                tool_call_id="search",
            ),
        ],
    )

    async def handler(request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            result=[
                AIMessage(
                    content=f"Answer\n\n**Relevant docs:**\n- [Guide]({footer_url})"
                )
            ]
        )

    async def check_urls(urls: list[str], timeout: float):
        checked_urls.extend(urls)
        return [LinkCheckResult(url=url, valid=True) for url in urls]

    monkeypatch.setattr(citation_module, "_check_urls_async", check_urls)
    result = asyncio.run(middleware.awrap_model_call(request, handler))

    assert checked_urls == [footer_url]
    assert footer_url in result.result[0].content


def test_trailing_slash_variant_accepts_grounded_footer_url():
    from src.middleware.citation_guard_middleware import CitationGuardMiddleware

    middleware = CitationGuardMiddleware()
    messages = [
        ToolMessage(
            content="Retrieved URL: https://docs.langchain.com/guide",
            name="search_docs_by_lang_chain",
            tool_call_id="search",
        ),
        ToolMessage(
            content="Valid links:\n- https://docs.langchain.com/guide",
            name="check_links",
            tool_call_id="check",
        ),
    ]
    response = ModelResponse(
        result=[
            AIMessage(
                content=(
                    "Answer\n\n**Relevant docs:**\n"
                    "- [Guide](https://docs.langchain.com/guide/)"
                )
            )
        ]
    )

    footer, invalid_urls = asyncio.run(
        middleware._invalid_footer_urls(response, messages)
    )

    assert footer is not None
    assert invalid_urls == set()


def test_unrelated_footer_url_is_rejected():
    from src.middleware.citation_guard_middleware import CitationGuardMiddleware

    middleware = CitationGuardMiddleware()
    unrelated_url = "https://example.com/unrelated"
    messages = [
        ToolMessage(
            content="Retrieved URL: https://docs.langchain.com/guide",
            name="search_docs_by_lang_chain",
            tool_call_id="search",
        )
    ]
    response = ModelResponse(
        result=[
            AIMessage(
                content=(f"Answer\n\n**Relevant docs:**\n- [Guide]({unrelated_url})")
            )
        ]
    )

    footer, invalid_urls = asyncio.run(
        middleware._invalid_footer_urls(response, messages)
    )

    assert footer is not None
    assert invalid_urls == {unrelated_url}
