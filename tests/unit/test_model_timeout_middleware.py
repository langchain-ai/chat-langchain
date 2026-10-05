import asyncio

import pytest
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse

from src.middleware.retry_middleware import ModelTimeoutMiddleware


class FakeModel:
    _llm_type = "fake-chat"


def _request(model):
    return ModelRequest(model=model, messages=[])


def test_model_timeout_middleware_raises_timeout_error():
    middleware = ModelTimeoutMiddleware(timeout=0.01)

    async def handler(request):  # noqa: ARG001
        await asyncio.sleep(1)

    with pytest.raises(TimeoutError, match="Model call timed out"):
        asyncio.run(middleware.awrap_model_call(_request(FakeModel()), handler))


def test_model_timeout_allows_fallback_after_primary_timeout():
    primary = FakeModel()
    fallback = FakeModel()
    timeout_middleware = ModelTimeoutMiddleware(timeout=0.01)
    fallback_middleware = ModelFallbackMiddleware.__new__(ModelFallbackMiddleware)
    fallback_middleware.models = [fallback]

    async def handler(request):
        if request.model is primary:
            await asyncio.sleep(1)
        return ModelResponse(result=[])

    result = asyncio.run(
        fallback_middleware.awrap_model_call(
            _request(primary),
            lambda request: timeout_middleware.awrap_model_call(request, handler),
        )
    )

    assert result.result == []
