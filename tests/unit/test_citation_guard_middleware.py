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
    assert "Relevant docs:" not in result.result[0].content


def test_remove_footer_urls_keeps_grounded_urls():
    from src.middleware.citation_guard_middleware import CitationGuardMiddleware

    valid_url = "https://docs.langchain.com/guide"
    invalid_url = "https://docs.langchain.com/invented"
    text = (
        f"Answer\n\n**Relevant docs:**\n- [Guide]({valid_url})\n"
        f"- [Invented]({invalid_url})"
    )

    result = CitationGuardMiddleware()._remove_footer_urls(text, {invalid_url})

    assert result == f"Answer\n\n**Relevant docs:**\n- [Guide]({valid_url})"


def test_remove_footer_urls_drops_empty_footer():
    from src.middleware.citation_guard_middleware import CitationGuardMiddleware

    invalid_url = "https://docs.langchain.com/invented"
    text = f"Answer\n\n**Relevant docs:**\n- [Invented]({invalid_url})"

    result = CitationGuardMiddleware()._remove_footer_urls(text, {invalid_url})

    assert result == "Answer"
    assert "Relevant docs:" not in result
