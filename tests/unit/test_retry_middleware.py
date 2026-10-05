import asyncio

import pytest
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from src.middleware.retry_middleware import ModelRetryMiddleware


def _request(model):
    return ModelRequest(model=model, messages=[HumanMessage(content="Hello")])


def test_timeout_retries_until_attempts_are_exhausted():
    """An indefinitely awaiting model call is retried and then raises."""
    model = FakeMessagesListChatModel(responses=[])
    middleware = ModelRetryMiddleware(
        max_retries=2,
        initial_delay=0,
        timeout_seconds=0.01,
    )
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        await asyncio.Event().wait()

    with pytest.raises(TimeoutError):
        asyncio.run(middleware.awrap_model_call(_request(model), handler))

    assert calls == 3


def test_fallback_runs_after_timed_out_primary_retries():
    """Fallback receives control after primary timeout retries are exhausted."""
    primary = FakeMessagesListChatModel(responses=[])
    fallback = FakeMessagesListChatModel(
        responses=[AIMessage(content="fallback response")]
    )
    retry = ModelRetryMiddleware(max_retries=1, initial_delay=0, timeout_seconds=0.01)
    fallback_middleware = ModelFallbackMiddleware(fallback)
    primary_calls = 0

    async def model_handler(request):
        nonlocal primary_calls
        if request.model is primary:
            primary_calls += 1
            await asyncio.Event().wait()
        return ModelResponse(result=[AIMessage(content="fallback response")])

    async def retry_handler(request):
        return await retry.awrap_model_call(request, model_handler)

    result = asyncio.run(
        fallback_middleware.awrap_model_call(_request(primary), retry_handler)
    )

    assert result.result[0].content == "fallback response"
    assert primary_calls == 2
