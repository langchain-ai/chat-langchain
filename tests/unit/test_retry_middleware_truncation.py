"""Tests for finish_reason=length truncation handling in ModelRetryMiddleware."""

from __future__ import annotations

import os

import pytest
from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage

os.environ["USE_LOCAL_PROMPTS"] = "1"

from src.middleware.retry_middleware import (
    TRUNCATION_MARKER,
    ModelRetryMiddleware,
)

TRUNCATED_CODE_ANSWER = (
    "Here is an example:\n\n```python\n"
    "def fetch():\n    resp = requests.get(url)\n    return resp.jso"
)


def _response(message: AIMessage) -> ModelResponse:
    return ModelResponse(result=[message])


async def _handler_returning(response: ModelResponse):
    async def handler(_request):
        return response

    return handler


@pytest.mark.asyncio
async def test_length_finish_reason_appends_truncation_marker():
    middleware = ModelRetryMiddleware(max_retries=0)
    message = AIMessage(
        content=TRUNCATED_CODE_ANSWER,
        response_metadata={"finish_reason": "length"},
    )
    handler = await _handler_returning(_response(message))

    result = await middleware.awrap_model_call({}, handler)

    final = result.result[-1]
    assert final.content.endswith(TRUNCATION_MARKER)
    # After marking, the visible answer partitions into an even number of
    # triple-backtick fences (odd number of segments) — no dangling code block.
    assert len(final.content.split("```")) % 2 == 1


@pytest.mark.asyncio
async def test_no_marker_when_finish_reason_is_stop():
    middleware = ModelRetryMiddleware(max_retries=0)
    message = AIMessage(
        content="All done.",
        response_metadata={"finish_reason": "stop"},
    )
    handler = await _handler_returning(_response(message))

    result = await middleware.awrap_model_call({}, handler)

    assert result.result[-1].content == "All done."


@pytest.mark.asyncio
async def test_no_marker_when_tool_calls_present():
    middleware = ModelRetryMiddleware(max_retries=0)
    message = AIMessage(
        content="",
        response_metadata={"finish_reason": "length"},
        tool_calls=[{"name": "search", "args": {}, "id": "t1"}],
    )
    handler = await _handler_returning(_response(message))

    result = await middleware.awrap_model_call({}, handler)

    assert TRUNCATION_MARKER not in (result.result[-1].content or "")
