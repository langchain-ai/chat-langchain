"""Tests for ModelRetryMiddleware finish_reason handling."""

import asyncio

import pytest
from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage

from src.middleware.retry_middleware import (
    MalformedResponseError,
    ModelRetryMiddleware,
)


def _response(finish_reason: str | None) -> ModelResponse:
    metadata = {"finish_reason": finish_reason} if finish_reason else {}
    return ModelResponse(result=[AIMessage(content="", response_metadata=metadata)])


def test_malformed_function_call_is_retried_then_succeeds():
    middleware = ModelRetryMiddleware(max_retries=2, initial_delay=0)
    responses = [_response("MALFORMED_FUNCTION_CALL"), _response("STOP")]
    calls = []

    async def handler(request):
        calls.append(request)
        return responses[len(calls) - 1]

    result = asyncio.run(middleware.awrap_model_call(object(), handler))

    assert len(calls) == 2
    assert result.result[-1].response_metadata["finish_reason"] == "STOP"


def test_malformed_function_call_retries_up_to_max_retries():
    middleware = ModelRetryMiddleware(max_retries=1, initial_delay=0)
    calls = []

    async def handler(request):
        calls.append(request)
        return _response("MALFORMED_FUNCTION_CALL")

    # The final attempt is returned as-is (existing behaviour); only the
    # retries before it are asserted here.
    asyncio.run(middleware.awrap_model_call(object(), handler))
    assert len(calls) == 2


def test_get_finish_reason_reads_last_ai_message():
    middleware = ModelRetryMiddleware()
    assert (
        middleware._get_finish_reason(_response("MALFORMED_FUNCTION_CALL"))
        == "MALFORMED_FUNCTION_CALL"
    )
    assert middleware._get_finish_reason(_response(None)) == ""
