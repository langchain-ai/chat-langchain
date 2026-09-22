"""Tests for terminal documentation answer validation."""

import asyncio

from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage

from src.middleware.answer_sanity_guard_middleware import (
    AnswerSanityGuardMiddleware,
)


def _request() -> ModelRequest:
    return ModelRequest(
        model=object(), messages=[HumanMessage(content="How do I use LangSmith?")]
    )


def test_clean_prose_with_footer_passes_untouched():
    middleware = AnswerSanityGuardMiddleware()
    response = ModelResponse(
        result=[
            AIMessage(
                content="Use the tracing client to record runs.\n\nRelevant docs: https://docs.example.com"
            )
        ]
    )
    calls = []

    async def handler(request):
        calls.append(request)
        return response

    result = asyncio.run(middleware.awrap_model_call(_request(), handler))

    assert result is response
    assert len(calls) == 1


def test_response_with_pending_tool_calls_passes_untouched():
    middleware = AnswerSanityGuardMiddleware()
    response = ModelResponse(
        result=[
            AIMessage(
                content="<Tabs>source</Tabs>",
                tool_calls=[{"name": "search", "args": {}, "id": "call-1"}],
            )
        ]
    )
    calls = []

    async def handler(request):
        calls.append(request)
        return response

    result = asyncio.run(middleware.awrap_model_call(_request(), handler))

    assert result is response
    assert len(calls) == 1


def test_observed_failure_triggers_exactly_one_retry():
    middleware = AnswerSanityGuardMiddleware()
    responses = [
        ModelResponse(
            result=[
                AIMessage(
                    content=(
                        "ndexing-for-langsmith-evaluation\n<Tabs>source</Tabs>\n"
                        "Edit this page on GitHub"
                    )
                )
            ]
        ),
        ModelResponse(result=[AIMessage(content="The answer is available in prose.")]),
    ]
    calls = []

    async def handler(request):
        calls.append(request)
        return responses.pop(0)

    result = asyncio.run(middleware.awrap_model_call(_request(), handler))

    assert len(calls) == 2
    assert "retrieved documentation" in calls[1].messages[-1].content
    assert result.result[0].content == "The answer is available in prose."


def test_good_retry_is_returned():
    middleware = AnswerSanityGuardMiddleware()
    responses = [
        ModelResponse(result=[AIMessage(content="<Tabs>bad</Tabs>")]),
        ModelResponse(
            result=[AIMessage(content="Use the client to trace requests.\n\nRelevant docs: https://docs.example.com")]
        ),
    ]

    async def handler(request):
        return responses.pop(0)

    result = asyncio.run(middleware.awrap_model_call(_request(), handler))

    assert "Use the client" in result.result[0].content


def test_second_degenerate_response_yields_apology():
    middleware = AnswerSanityGuardMiddleware()
    responses = [
        ModelResponse(result=[AIMessage(content="<Tabs>bad</Tabs>")]),
        ModelResponse(result=[AIMessage(content="--- stdout ---\nexit: 0")]),
    ]

    async def handler(request):
        return responses.pop(0)

    result = asyncio.run(middleware.awrap_model_call(_request(), handler))

    assert "rephrase" in result.result[0].content


def test_closed_code_block_and_cjk_sentence_are_not_flagged():
    middleware = AnswerSanityGuardMiddleware()

    for content in (
        "Use this example:\n```python\nprint('hello')\n```",
        "这是一个关于如何使用 LangSmith 的完整说明，能够正常回答用户的问题。",
    ):
        response = ModelResponse(result=[AIMessage(content=content)])
        calls = []

        async def handler(request):
            calls.append(request)
            return response

        result = asyncio.run(middleware.awrap_model_call(_request(), handler))

        assert result is response
        assert len(calls) == 1
