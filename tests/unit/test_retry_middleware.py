import asyncio

import pytest
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from src.middleware.retry_middleware import (
    ModelRetryMiddleware,
    _ProviderValidationAwareRunnableRetry,
)


class _FallbackModel(BaseChatModel):
    model_name: str

    @property
    def _llm_type(self) -> str:
        return "test-fallback"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="answer"))]
        )

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="answer"))]
        )


def test_provider_validation_error_is_not_retried():
    middleware = ModelRetryMiddleware(max_retries=2, initial_delay=0)
    calls = 0

    async def handler(request: ModelRequest):
        nonlocal calls
        calls += 1
        raise ValueError("unsupported request shape")

    with pytest.raises(ValueError, match="unsupported request shape"):
        asyncio.run(
            middleware.awrap_model_call(
                ModelRequest(model=object(), messages=[HumanMessage(content="Hi")]),
                handler,
            )
        )

    assert calls == 1


def test_model_retry_wrapper_does_not_retry_provider_validation_error():
    calls = 0

    def invoke(_input):
        nonlocal calls
        calls += 1
        raise ValueError("unsupported request shape")

    runnable = _ProviderValidationAwareRunnableRetry(
        bound=RunnableLambda(invoke), max_attempt_number=3
    )

    with pytest.raises(ValueError, match="unsupported request shape"):
        runnable.invoke("request")

    assert calls == 1


def test_model_retry_times_out_hung_handler():
    async def run_test():
        middleware = ModelRetryMiddleware(
            max_retries=1,
            initial_delay=0,
            call_timeout=0.01,
        )
        calls = 0

        async def handler(request: ModelRequest):
            nonlocal calls
            calls += 1
            await asyncio.Event().wait()

        with pytest.raises(asyncio.TimeoutError):
            await middleware.awrap_model_call(
                ModelRequest(model=_FallbackModel(model_name="primary"), messages=[]),
                handler,
            )

        assert calls == 2

    asyncio.run(run_test())


def test_model_fallback_recovers_after_primary_timeout():
    async def run_test():
        primary = _FallbackModel(model_name="primary")
        fallback = _FallbackModel(model_name="fallback")
        retry = ModelRetryMiddleware(max_retries=0, call_timeout=0.01)
        fallback_middleware = ModelFallbackMiddleware(fallback)
        request = ModelRequest(model=primary, messages=[HumanMessage(content="Hi")])

        async def model_handler(current_request: ModelRequest):
            if current_request.model is primary:
                await asyncio.Event().wait()
            return ModelResponse(result=[AIMessage(content="fallback answer")])

        async def retry_handler(current_request: ModelRequest):
            return await retry.awrap_model_call(current_request, model_handler)

        response = await fallback_middleware.awrap_model_call(request, retry_handler)

        assert response.result[0].content == "fallback answer"

    asyncio.run(run_test())
