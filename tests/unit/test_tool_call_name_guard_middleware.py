"""Tests for malformed model-emitted tool-call names."""

import asyncio

from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool

from src.middleware.tool_call_name_guard_middleware import (
    ToolCallNameGuardMiddleware,
)


def _check_links_tool():
    return StructuredTool.from_function(
        lambda url: url,
        name="check_links",
        description="Check links.",
    )


def test_malformed_name_is_rewritten_and_args_preserved():
    middleware = ToolCallNameGuardMiddleware()
    malformed_name = (
        "check_filesize??  if present or not, wait, not a tool ... "
        "call:default_api:check_links"
    )
    request = ModelRequest(
        model=object(),
        messages=[HumanMessage(content="Check this URL.")],
        tools=[_check_links_tool()],
    )

    async def handler(request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            result=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": malformed_name,
                            "args": {"url": "https://example.com"},
                            "id": "call-1",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        )

    response = asyncio.run(middleware.awrap_model_call(request, handler))

    assert response.result[0].tool_calls[0]["name"] == "check_links"
    assert response.result[0].tool_calls[0]["args"] == {"url": "https://example.com"}


def test_unrecoverable_name_regenerates_once():
    middleware = ToolCallNameGuardMiddleware()
    calls = 0
    request = ModelRequest(
        model=object(),
        messages=[HumanMessage(content="Check this URL.")],
        tools=[_check_links_tool()],
    )

    async def handler(request: ModelRequest) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(
                result=[
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "not-a-registered-tool\n",
                                "args": {},
                                "id": "call-1",
                                "type": "tool_call",
                            }
                        ],
                    )
                ]
            )
        return ModelResponse(result=[AIMessage(content="Recovered answer.")])

    response = asyncio.run(middleware.awrap_model_call(request, handler))

    assert calls == 2
    assert response.result[0].content == "Recovered answer."


def test_well_formed_name_passes_through_without_retry():
    middleware = ToolCallNameGuardMiddleware()
    calls = 0
    request = ModelRequest(
        model=object(),
        messages=[HumanMessage(content="Check this URL.")],
        tools=[_check_links_tool()],
    )

    async def handler(request: ModelRequest) -> ModelResponse:
        nonlocal calls
        calls += 1
        return ModelResponse(
            result=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "check_links",
                            "args": {"url": "https://example.com"},
                            "id": "call-1",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        )

    response = asyncio.run(middleware.awrap_model_call(request, handler))

    assert calls == 1
    assert response.result[0].tool_calls[0]["name"] == "check_links"


def test_poisoned_history_is_sanitized_before_fallback_handler():
    middleware = ToolCallNameGuardMiddleware()
    seen_requests: list[ModelRequest] = []
    request = ModelRequest(
        model=object(),
        messages=[
            HumanMessage(content="Continue."),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "x" * 114,
                        "args": {},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
        ],
        tools=[_check_links_tool()],
    )

    async def handler(request: ModelRequest) -> ModelResponse:
        seen_requests.append(request)
        return ModelResponse(result=[AIMessage(content="Fallback answer.")])

    response = asyncio.run(middleware.awrap_model_call(request, handler))

    assert response.result[0].content == "Fallback answer."
    assert len(seen_requests) == 1
    assert seen_requests[0].messages[1].tool_calls == []
