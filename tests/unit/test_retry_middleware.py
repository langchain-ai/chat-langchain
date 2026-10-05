import asyncio

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from src.middleware.retry_middleware import (
    ModelRetryMiddleware,
    PermanentRequestAwareModelFallbackMiddleware,
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


def test_model_retry_middleware_does_not_retry_400_error():
    middleware = ModelRetryMiddleware(max_retries=2, initial_delay=0)
    calls = 0

    class ProviderRequestError(Exception):
        status_code = 400

    async def handler(request: ModelRequest):
        nonlocal calls
        calls += 1
        raise ProviderRequestError("invalid tool choice")

    with pytest.raises(ProviderRequestError, match="invalid tool choice"):
        asyncio.run(
            middleware.awrap_model_call(
                ModelRequest(model=object(), messages=[HumanMessage(content="Hi")]),
                handler,
            )
        )

    assert calls == 1


def test_model_retry_middleware_retries_transient_error():
    middleware = ModelRetryMiddleware(max_retries=2, initial_delay=0)
    calls = 0

    async def handler(request: ModelRequest):
        nonlocal calls
        calls += 1
        raise TimeoutError("provider timeout")

    with pytest.raises(TimeoutError, match="provider timeout"):
        asyncio.run(
            middleware.awrap_model_call(
                ModelRequest(model=object(), messages=[HumanMessage(content="Hi")]),
                handler,
            )
        )

    assert calls == 3


class _FallbackModel:
    _llm_type = "test"


def test_fallback_middleware_skips_permanent_error():
    middleware = PermanentRequestAwareModelFallbackMiddleware(
        _FallbackModel(), _FallbackModel()
    )
    calls = 0

    class BadRequestError(Exception):
        pass

    async def handler(request: ModelRequest):
        nonlocal calls
        calls += 1
        raise BadRequestError("invalid_request_error")

    with pytest.raises(BadRequestError, match="invalid_request_error"):
        asyncio.run(
            middleware.awrap_model_call(
                ModelRequest(
                    model=_FallbackModel(), messages=[HumanMessage(content="Hi")]
                ),
                handler,
            )
        )

    assert calls == 1


def test_fallback_middleware_uses_fallback_for_transient_error():
    primary_model = _FallbackModel()
    fallback_model = _FallbackModel()
    middleware = PermanentRequestAwareModelFallbackMiddleware(fallback_model)
    calls = 0

    async def handler(request: ModelRequest):
        nonlocal calls
        calls += 1
        if request.model is primary_model:
            raise TimeoutError("provider timeout")
        return ModelResponse(result=[AIMessage(content="fallback")])

    result = asyncio.run(
        middleware.awrap_model_call(
            ModelRequest(model=primary_model, messages=[HumanMessage(content="Hi")]),
            handler,
        )
    )

    assert result.result[0].content == "fallback"
    assert calls == 2
