"""Tests for fresh documentation research on each technical turn."""

import asyncio

from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.middleware.docs_research_guard_middleware import DocsResearchGuardMiddleware


def test_greeting_with_product_names_does_not_force_research():
    middleware = DocsResearchGuardMiddleware()
    request = ModelRequest(
        model=object(),
        messages=[HumanMessage(content="Hi")],
    )
    response = ModelResponse(
        result=[
            AIMessage(
                content=(
                    "Hello! How can I help you with LangChain, LangGraph, "
                    "LangSmith, Fleet, or DeepAgents today?"
                )
            )
        ]
    )

    assert not middleware._should_retry(request, response)


def test_non_latin_greeting_does_not_force_research():
    middleware = DocsResearchGuardMiddleware()
    request = ModelRequest(
        model=object(),
        messages=[HumanMessage(content="こんにちは")],
    )
    response = ModelResponse(result=[AIMessage(content="こんにちは！")])

    assert not middleware._should_retry(request, response)


def test_technical_answers_still_force_research():
    middleware = DocsResearchGuardMiddleware()
    request = ModelRequest(
        model=object(),
        messages=[HumanMessage(content="How do I configure this?")],
    )

    for answer in (
        "The StateGraph class accepts a config_schema parameter for this setup.",
        "Use the config_schema parameter to configure the graph before invoking it.",
        "Create the graph like this:\n```python\ngraph = StateGraph()\n```",
    ):
        response = ModelResponse(result=[AIMessage(content=answer)])
        assert middleware._should_retry(request, response)


def test_user_turn_signal_detects_technical_question():
    middleware = DocsResearchGuardMiddleware()

    assert middleware._user_turn_has_technical_signal(
        HumanMessage(content="How do I use StateGraph with a config_schema parameter?")
    )


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
    assert calls[1].messages[:-1] == messages
    assert isinstance(calls[1].messages[-1], HumanMessage)
    assert "research this question on this turn" in calls[1].messages[-1].content
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

    assert len(calls) == 3


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
    assert invented not in result.result[0].content[0]["text"]
    assert grounded in result.result[0].content[0]["text"]


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


def test_pricing_footer_url_is_grounded_without_retry(monkeypatch):
    from src.middleware import citation_guard_middleware as citation_module
    from src.middleware.citation_guard_middleware import CitationGuardMiddleware
    from src.tools.link_check_tools import LinkCheckResult

    url = "https://www.langchain.com/pricing"
    checks: list[list[str]] = []
    middleware = CitationGuardMiddleware()
    request = ModelRequest(
        model=object(),
        messages=[
            HumanMessage(content="What are the LangChain plans?"),
            ToolMessage(
                content=f"Source: {url}\n\nPricing details",
                name="fetch_langchain_pricing",
                tool_call_id="pricing",
            ),
        ],
    )
    response = ModelResponse(
        result=[
            AIMessage(content=f"**Answer**\n\n**Relevant docs:**\n- [Pricing]({url})")
        ]
    )

    async def handler(request: ModelRequest) -> ModelResponse:
        return response

    async def check_urls(urls: list[str], timeout: float):
        checks.append(urls)
        return [LinkCheckResult(url=url, valid=True) for url in urls]

    monkeypatch.setattr(citation_module, "_check_urls_async", check_urls)
    result = asyncio.run(middleware.awrap_model_call(request, handler))

    assert result is response
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
    assert calls[1].messages[:-1] == request.messages
    assert isinstance(calls[1].messages[-1], HumanMessage)
    assert "Rewrite the Relevant docs footer" in calls[1].messages[-1].content
    assert result.result[0].content.startswith("**Answer**")


def test_list_content_rewrites_only_text_block_and_drops_invalid_url(monkeypatch):
    from src.middleware import citation_guard_middleware as citation_module
    from src.middleware.citation_guard_middleware import CitationGuardMiddleware

    valid_url = "https://docs.langchain.com/a"
    invalid_url = "https://bad.example/b"
    signature = "El4KXAER..."
    middleware = CitationGuardMiddleware()
    request = ModelRequest(
        model=object(),
        messages=[
            HumanMessage(content="Find docs."),
            ToolMessage(
                content=f"Retrieved URL: {valid_url}",
                name="search_docs_by_lang_chain",
                tool_call_id="search",
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
                                "Answer\n\n**Relevant docs:**\n"
                                f"- [A]({valid_url})\n- [B]({invalid_url})"
                            ),
                        },
                        {"thought_signature": signature},
                    ]
                )
            ]
        )

    async def check_urls(urls: list[str], timeout: float):
        from src.tools.link_check_tools import LinkCheckResult

        return [LinkCheckResult(url=url, valid=True) for url in urls]

    monkeypatch.setattr(citation_module, "_check_urls_async", check_urls)
    result = asyncio.run(middleware.awrap_model_call(request, handler))
    message = result.result[0]

    assert isinstance(message.content, list)
    assert invalid_url not in message.content[0]["text"]
    extracted_text = middleware._message_text(message)
    assert "text" not in extracted_text
    assert signature not in extracted_text


def test_plain_string_content_rewrites_footer():
    from src.middleware.citation_guard_middleware import CitationGuardMiddleware

    middleware = CitationGuardMiddleware()
    repaired = "Answer\n\n**Relevant docs:**"
    message = AIMessage(
        content="Answer\n\n**Relevant docs:**\n- [B](https://bad.example/b)"
    )
    response = ModelResponse(result=[message])

    middleware._replace_footer(response, message, repaired)

    assert response.result[0].content == repaired


def test_message_text_ignores_non_text_blocks():
    from src.middleware.citation_guard_middleware import CitationGuardMiddleware

    middleware = CitationGuardMiddleware()
    message = AIMessage(
        content=[
            {"type": "text", "text": "Answer"},
            {"thought_signature": "El4KXAER..."},
            {"type": "tool_use", "name": "check_links"},
        ]
    )

    assert middleware._message_text(message) == "Answer"


def test_search_results_satisfy_research_requirement():
    middleware = DocsResearchGuardMiddleware()
    calls: list[ModelRequest] = []
    request = ModelRequest(
        model=object(),
        messages=[
            HumanMessage(content="Explain StateGraph."),
            ToolMessage(
                content="StateGraph combines state and nodes.",
                name="search_docs_by_lang_chain",
                tool_call_id="search",
            ),
        ],
    )

    async def handler(request: ModelRequest) -> ModelResponse:
        calls.append(request)
        return ModelResponse(
            result=[
                AIMessage(
                    content="StateGraph is the graph class described in the docs."
                )
            ]
        )

    asyncio.run(middleware.awrap_model_call(request, handler))

    assert len(calls) == 1


def test_unread_large_result_pointer_does_not_satisfy_research_requirement():
    middleware = DocsResearchGuardMiddleware()
    calls: list[ModelRequest] = []
    request = ModelRequest(
        model=object(),
        messages=[
            HumanMessage(content="Explain StateGraph."),
            ToolMessage(
                content="/large_tool_results/result-123",
                name="search_docs_by_lang_chain",
                tool_call_id="search",
            ),
        ],
    )

    async def handler(request: ModelRequest) -> ModelResponse:
        calls.append(request)
        return ModelResponse(
            result=[
                AIMessage(
                    content="StateGraph is the graph class described in the docs."
                )
            ]
        )

    result = asyncio.run(middleware.awrap_model_call(request, handler))

    assert len(calls) == 3
    assert result.result[0].content.startswith("Documentation could not be consulted")


def test_check_links_only_is_not_research():
    middleware = DocsResearchGuardMiddleware()
    calls: list[ModelRequest] = []
    request = ModelRequest(
        model=object(),
        messages=[
            HumanMessage(content="Explain the StateGraph constructor."),
            ToolMessage(
                content="Link is valid", name="check_links", tool_call_id="link"
            ),
        ],
    )

    async def handler(request: ModelRequest) -> ModelResponse:
        calls.append(request)
        return ModelResponse(
            result=[
                AIMessage(
                    content="The StateGraph constructor accepts configuration options."
                )
            ]
        )

    result = asyncio.run(middleware.awrap_model_call(request, handler))

    assert len(calls) == 3
    assert result.result[0].content.startswith("Documentation could not be consulted")


def test_ignored_retries_are_bounded_and_sanitized():
    middleware = DocsResearchGuardMiddleware()
    calls: list[ModelRequest] = []
    url = "https://docs.langchain.com/oss/python/langgraph/graph-api"
    request = ModelRequest(
        model=object(),
        messages=[HumanMessage(content="How do I build a graph?")],
    )

    async def handler(request: ModelRequest) -> ModelResponse:
        calls.append(request)
        return ModelResponse(
            result=[
                AIMessage(
                    content=f"Use this API:\n\n```python\nStateGraph()\n```\n\nSee {url}."
                )
            ]
        )

    result = asyncio.run(middleware.awrap_model_call(request, handler))

    assert len(calls) == 3
    assert all(call.tool_choice for call in calls[1:])
    assert "```" not in result.result[0].content
    assert url not in result.result[0].content
    assert result.result[0].content.startswith("Documentation could not be consulted")


def test_disabled_env_escape_hatch_skips_enforcement(monkeypatch):
    monkeypatch.setenv("DOCS_RESEARCH_GUARD_DISABLED", "true")
    middleware = DocsResearchGuardMiddleware()
    calls: list[ModelRequest] = []
    request = ModelRequest(
        model=object(),
        messages=[HumanMessage(content="Explain the StateGraph constructor.")],
    )

    async def handler(request: ModelRequest) -> ModelResponse:
        calls.append(request)
        return ModelResponse(result=[AIMessage(content="```python\nStateGraph()\n```")])

    result = asyncio.run(middleware.awrap_model_call(request, handler))

    assert len(calls) == 1
    assert "```" in result.result[0].content
