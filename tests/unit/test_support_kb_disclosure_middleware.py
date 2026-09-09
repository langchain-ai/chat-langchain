"""Tests for support knowledge base outage disclosure."""

import asyncio

from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.middleware.support_kb_disclosure_middleware import (
    SupportKBDisclosureMiddleware,
)


def test_all_support_kb_errors_force_disclosure():
    middleware = SupportKBDisclosureMiddleware()
    calls: list[ModelRequest] = []
    request = ModelRequest(
        model=object(),
        messages=[
            HumanMessage(content="How do I reset my subscription?"),
            ToolMessage(
                content="unavailable",
                name="search_support_articles",
                tool_call_id="search",
                status="error",
            ),
            ToolMessage(
                content="unavailable",
                name="get_support_article_content",
                tool_call_id="content",
                status="error",
            ),
        ],
    )

    async def handler(request: ModelRequest) -> ModelResponse:
        calls.append(request)
        return ModelResponse(result=[AIMessage(content="Here is the support policy.")])

    asyncio.run(middleware.awrap_model_call(request, handler))

    assert len(calls) == 2
    assert "support knowledge base could not be consulted" in calls[1].system_prompt


def test_any_successful_support_kb_call_does_not_force_disclosure():
    middleware = SupportKBDisclosureMiddleware()
    calls: list[ModelRequest] = []
    request = ModelRequest(
        model=object(),
        messages=[
            HumanMessage(content="How do I reset my subscription?"),
            ToolMessage(
                content="unavailable",
                name="search_support_articles",
                tool_call_id="search-success-case",
                status="error",
            ),
            ToolMessage(
                content="article",
                name="get_support_article_content",
                tool_call_id="content-success-case",
                status="success",
            ),
        ],
    )

    async def handler(request: ModelRequest) -> ModelResponse:
        calls.append(request)
        return ModelResponse(result=[AIMessage(content="Here is the support policy.")])

    asyncio.run(middleware.awrap_model_call(request, handler))

    assert len(calls) == 1
