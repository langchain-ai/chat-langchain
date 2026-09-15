"""Tests for support knowledge-base outage disclosure."""

import asyncio

from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.middleware.support_kb_disclosure_middleware import (
    DISCLOSURE,
    SupportKbDisclosureMiddleware,
)


def _request(*messages):
    return ModelRequest(model=object(), messages=list(messages))


def _failed_tool():
    return ToolMessage(
        content="PylonUnavailableError: Pylon API returned HTTP 401",
        name="search_support_articles",
        tool_call_id="support-search",
        status="error",
    )


def test_failed_support_kb_forces_exactly_one_retry():
    middleware = SupportKbDisclosureMiddleware()
    calls = []

    async def handler(request):
        calls.append(request)
        return ModelResponse(result=[AIMessage(content="A" * 140)])

    response = asyncio.run(
        middleware.awrap_model_call(
            _request(HumanMessage(content="How does retention work?"), _failed_tool()),
            handler,
        )
    )

    assert len(calls) == 2
    assert DISCLOSURE in response.result[0].content
    assert DISCLOSURE in calls[1].system_prompt


def test_existing_disclosure_passes_through_untouched():
    middleware = SupportKbDisclosureMiddleware()
    calls = []
    answer = f"{'A' * 140} {DISCLOSURE}"

    async def handler(request):
        calls.append(request)
        return ModelResponse(result=[AIMessage(content=answer)])

    response = asyncio.run(
        middleware.awrap_model_call(
            _request(HumanMessage(content="How does retention work?"), _failed_tool()),
            handler,
        )
    )

    assert len(calls) == 1
    assert response.result[0].content == answer


def test_no_support_kb_failure_passes_through_untouched():
    middleware = SupportKbDisclosureMiddleware()
    calls = []

    async def handler(request):
        calls.append(request)
        return ModelResponse(result=[AIMessage(content="A" * 140)])

    response = asyncio.run(
        middleware.awrap_model_call(
            _request(
                HumanMessage(content="How does retention work?"),
                ToolMessage(
                    content="No results found",
                    name="search_support_articles",
                    tool_call_id="support-search",
                ),
            ),
            handler,
        )
    )

    assert len(calls) == 1
    assert response.result[0].content == "A" * 140


def test_short_greeting_passes_through_untouched():
    middleware = SupportKbDisclosureMiddleware()
    calls = []

    async def handler(request):
        calls.append(request)
        return ModelResponse(result=[AIMessage(content="Hello!")])

    response = asyncio.run(
        middleware.awrap_model_call(
            _request(HumanMessage(content="Hi"), _failed_tool()),
            handler,
        )
    )

    assert len(calls) == 1
    assert response.result[0].content == "Hello!"
