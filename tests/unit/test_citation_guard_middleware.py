import asyncio

from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


def test_anchored_documentation_url_grounds_base_footer_without_retry(monkeypatch):
    from src.middleware import citation_guard_middleware as citation_module
    from src.middleware.citation_guard_middleware import CitationGuardMiddleware

    base_url = "https://docs.langchain.com/oss/python/langgraph/graph-api"
    calls: list[ModelRequest] = []
    checks: list[list[str]] = []
    middleware = CitationGuardMiddleware()
    request = ModelRequest(
        model=object(),
        messages=[
            HumanMessage(content="How does graph invocation work?"),
            ToolMessage(
                content="https://docs.langchain.com/oss/python/langgraph/graph-api#input-to-invoke-or-stream",
                name="search_docs_by_lang_chain",
                tool_call_id="search",
            ),
        ],
    )

    async def handler(request: ModelRequest) -> ModelResponse:
        calls.append(request)
        return ModelResponse(
            result=[AIMessage(content=f"Answer\n\n**Relevant docs:**\n- [Guide]({base_url})")]
        )

    async def check_urls(urls: list[str], timeout: float):
        checks.append(urls)
        from src.tools.link_check_tools import LinkCheckResult

        return [LinkCheckResult(url=url, valid=True) for url in urls]

    monkeypatch.setattr(citation_module, "_check_urls_async", check_urls)
    result = asyncio.run(middleware.awrap_model_call(request, handler))

    assert len(calls) == 1
    assert checks == [[base_url]]
    assert base_url in result.result[0].content


def test_check_links_valid_url_is_accepted_without_documentation_url(monkeypatch):
    from src.middleware import citation_guard_middleware as citation_module
    from src.middleware.citation_guard_middleware import CitationGuardMiddleware

    url = "https://docs.langchain.com/oss/python/langgraph/graph-api/"
    calls: list[ModelRequest] = []
    middleware = CitationGuardMiddleware()
    request = ModelRequest(
        model=object(),
        messages=[
            HumanMessage(content="How does graph invocation work?"),
            ToolMessage(
                content="/large_tool_results/docs-page.txt",
                name="query_docs_filesystem_docs_by_lang_chain",
                tool_call_id="read",
            ),
            ToolMessage(
                content=f"Valid links:\n- {url}",
                name="check_links",
                tool_call_id="check",
            ),
        ],
    )

    async def handler(request: ModelRequest) -> ModelResponse:
        calls.append(request)
        return ModelResponse(
            result=[AIMessage(content=f"Answer\n\n**Relevant docs:**\n- [Guide]({url})")]
        )

    async def check_urls(urls: list[str], timeout: float):
        raise AssertionError("check_links-valid URL should not be rechecked")

    monkeypatch.setattr(citation_module, "_check_urls_async", check_urls)
    result = asyncio.run(middleware.awrap_model_call(request, handler))

    assert len(calls) == 1
    assert url in result.result[0].content


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
    second_result = asyncio.run(middleware.awrap_model_call(request, handler))

    assert len(calls) == 3
    assert retry_url not in result.result[0].content
    assert result.result[0].content == "Answer\n\n**Relevant docs:**"
    assert second_result.result[0].content == "Answer\n\n**Relevant docs:**"
