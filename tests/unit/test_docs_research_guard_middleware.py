"""Tests for fresh documentation research on each technical turn."""

import asyncio

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
