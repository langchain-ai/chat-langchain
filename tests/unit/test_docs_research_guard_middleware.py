"""Tests for fresh documentation research on each technical turn."""

import asyncio

from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.middleware.docs_research_guard_middleware import DocsResearchGuardMiddleware


def test_oversized_retrieval_result_is_bounded_before_model_call():
    middleware = DocsResearchGuardMiddleware(retrieval_result_char_limit=100)
    calls: list[ModelRequest] = []
    messages = [
        HumanMessage(content="Find the docs."),
        ToolMessage(
            content="title: StateGraph\n\n" + ("documentation " * 100),
            name="query_docs_filesystem_docs_by_lang_chain",
            tool_call_id="filesystem-read",
        ),
    ]

    async def handler(request: ModelRequest) -> ModelResponse:
        calls.append(request)
        return ModelResponse(result=[AIMessage(content="Done.")])

    asyncio.run(middleware.awrap_model_call(ModelRequest(model=object(), messages=messages), handler))

    result = calls[0].messages[-1]
    assert isinstance(result, ToolMessage)
    assert len(result.content) <= 100
    assert "retrieval content omitted" in result.content
    assert result.name == "query_docs_filesystem_docs_by_lang_chain"
    assert result.tool_call_id == "filesystem-read"


def test_oldest_retrieval_results_collapse_when_turn_budget_is_exceeded():
    middleware = DocsResearchGuardMiddleware(
        retrieval_result_char_limit=10_000,
        retrieval_turn_char_limit=8_000,
    )
    calls: list[ModelRequest] = []
    first = "first result\n" + ("a" * 4_500)
    second = "second result\n" + ("b" * 4_500)
    third = "third result\n" + ("c" * 4_500)
    messages = [
        HumanMessage(content="Find the docs."),
        ToolMessage(content=first, name="search_docs_by_lang_chain", tool_call_id="1"),
        ToolMessage(content=second, name="search_docs_by_lang_chain", tool_call_id="2"),
        ToolMessage(content=third, name="search_docs_by_lang_chain", tool_call_id="3"),
    ]

    async def handler(request: ModelRequest) -> ModelResponse:
        calls.append(request)
        return ModelResponse(result=[AIMessage(content="Done.")])

    asyncio.run(middleware.awrap_model_call(ModelRequest(model=object(), messages=messages), handler))

    bounded = calls[0].messages[1:]
    assert "retrieval content omitted" in bounded[0].content
    assert "retrieval content omitted" in bounded[1].content
    assert bounded[2].content == third
    assert sum(len(message.content) for message in bounded) <= 8_000


def test_small_retrieval_turn_remains_unchanged():
    middleware = DocsResearchGuardMiddleware()
    calls: list[ModelRequest] = []
    messages = [
        HumanMessage(content="Find the docs."),
        ToolMessage(
            content="small result",
            name="search_docs_by_lang_chain",
            tool_call_id="small-search",
        ),
    ]

    async def handler(request: ModelRequest) -> ModelResponse:
        calls.append(request)
        return ModelResponse(result=[AIMessage(content="Done.")])

    asyncio.run(middleware.awrap_model_call(ModelRequest(model=object(), messages=messages), handler))

    assert calls[0].messages == messages


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
                result=[AIMessage(content="StateGraph accepts the configSchema option.")]
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
    assert calls[1].messages[-1].content == "StateGraph accepts the configSchema option."
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
