import asyncio
import threading

import pytest
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from src.middleware.retry_middleware import ModelRetryMiddleware


def _request(model):
    return ModelRequest(model=model, messages=[HumanMessage(content="hello")])


def test_async_model_timeout_is_retryable():
    async def run_test():
        middleware = ModelRetryMiddleware(max_retries=0, timeout_seconds=0.01)
        model = FakeMessagesListChatModel(responses=[])
        never_returns = asyncio.Event()

        async def handler(request):
            await never_returns.wait()
            return ModelResponse(result=[AIMessage(content="unreachable")])

        with pytest.raises(asyncio.TimeoutError):
            await middleware.awrap_model_call(_request(model), handler)

    asyncio.run(run_test())


def test_async_model_timeout_uses_fallback():
    async def run_test():
        primary = FakeMessagesListChatModel(responses=[])
        fallback = FakeMessagesListChatModel(
            responses=[AIMessage(content="fallback answer")]
        )
        retry = ModelRetryMiddleware(max_retries=0, timeout_seconds=0.01)
        fallback_middleware = ModelFallbackMiddleware(primary, fallback)
        never_returns = asyncio.Event()

        async def handler(request):
            if request.model is primary:
                await never_returns.wait()
            return ModelResponse(result=[AIMessage(content="fallback answer")])

        async def retrying_handler(request):
            return await retry.awrap_model_call(request, handler)

        response = await fallback_middleware.awrap_model_call(
            _request(primary), retrying_handler
        )

        assert response.result[0].content == "fallback answer"

    asyncio.run(run_test())


def test_sync_model_timeout_is_retryable():
    middleware = ModelRetryMiddleware(max_retries=0, timeout_seconds=0.01)
    model = FakeMessagesListChatModel(responses=[])

    def handler(request):
        threading.Event().wait()
        return ModelResponse(result=[AIMessage(content="unreachable")])

    with pytest.raises(TimeoutError):
        middleware.wrap_model_call(_request(model), handler)
