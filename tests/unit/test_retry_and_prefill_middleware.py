from unittest.mock import MagicMock

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage

from src.middleware.prefill_middleware import RemoveTrailingAIMessageMiddleware
from src.middleware.retry_middleware import ModelRetryMiddleware


@pytest.mark.asyncio
async def test_prefill_value_error_is_attempted_once():
    middleware = ModelRetryMiddleware(max_retries=2, initial_delay=0)
    request = ModelRequest(model=MagicMock(), messages=[HumanMessage(content="hello")])
    attempts = 0

    async def handler(_request):
        nonlocal attempts
        attempts += 1
        raise ValueError("does not support model prefilling")

    with pytest.raises(ValueError, match="does not support model prefilling"):
        await middleware.awrap_model_call(request, handler)

    assert attempts == 1


def test_trailing_non_tool_call_ai_message_is_removed():
    middleware = RemoveTrailingAIMessageMiddleware()
    request = ModelRequest(
        model=MagicMock(),
        messages=[HumanMessage(content="hello"), AIMessage(content="draft")],
    )
    captured = []

    middleware.wrap_model_call(
        request, lambda updated: captured.append(updated) or MagicMock()
    )

    assert captured[0].messages == request.messages[:1]


def test_trailing_tool_call_ai_message_is_preserved():
    middleware = RemoveTrailingAIMessageMiddleware()
    tool_call = {"name": "search", "args": {}, "id": "call-1"}
    request = ModelRequest(
        model=MagicMock(),
        messages=[
            HumanMessage(content="hello"),
            AIMessage(content="", tool_calls=[tool_call]),
        ],
    )
    captured = []

    middleware.wrap_model_call(
        request, lambda updated: captured.append(updated) or MagicMock()
    )

    assert captured[0].messages == request.messages
