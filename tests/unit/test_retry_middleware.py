import asyncio

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda

from src.middleware.retry_middleware import (
    ModelRetryMiddleware,
    ProviderValidationAwareModelFallbackMiddleware,
    _ProviderValidationAwareRunnableRetry,
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


@pytest.mark.parametrize(
    "error_type",
    [
        type("BadRequestError", (Exception,), {"__module__": "openai"}),
        type("BadRequestError", (Exception,), {"__module__": "anthropic"}),
    ],
)
def test_provider_bad_request_does_not_retry_or_fallback(error_type):
    primary_model = object()
    fallback_model = object()
    fallback = ProviderValidationAwareModelFallbackMiddleware(fallback_model)
    retry = ModelRetryMiddleware(max_retries=2, initial_delay=0)
    calls = []

    async def provider_handler(request):
        calls.append(request.model)
        raise error_type("invalid request")

    async def retry_handler(request):
        return await retry.awrap_model_call(request, provider_handler)

    with pytest.raises(error_type):
        asyncio.run(
            fallback.awrap_model_call(
                ModelRequest(
                    model=primary_model, messages=[HumanMessage(content="Hi")]
                ),
                retry_handler,
            )
        )

    assert calls == [primary_model]


def test_transient_error_retries_and_falls_back():
    primary_model = object()
    fallback_model = object()
    fallback = ProviderValidationAwareModelFallbackMiddleware(fallback_model)
    retry = ModelRetryMiddleware(max_retries=2, initial_delay=0)
    calls = []

    async def provider_handler(request):
        calls.append(request.model)
        if request.model is primary_model:
            raise TimeoutError("temporary failure")
        return "success"

    async def retry_handler(request):
        return await retry.awrap_model_call(request, provider_handler)

    result = asyncio.run(
        fallback.awrap_model_call(
            ModelRequest(model=primary_model, messages=[HumanMessage(content="Hi")]),
            retry_handler,
        )
    )

    assert result == "success"
    assert calls == [primary_model, primary_model, primary_model, fallback_model]
