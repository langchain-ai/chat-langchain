import asyncio

from langchain.agents.middleware import (
    ModelFallbackMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage

from src.middleware.model_timeout_middleware import ModelCallTimeoutMiddleware


def _request(model: object = None) -> ModelRequest:
    return ModelRequest(model=model, messages=[])


def test_fast_model_call_passes_through_unchanged():
    middleware = ModelCallTimeoutMiddleware(timeout_s=0.1)
    response = ModelResponse([AIMessage(content="done")])

    async def handler(request):
        return response

    result = asyncio.run(middleware.awrap_model_call(_request(), handler))

    assert result is response


def test_stalled_model_call_raises_timeout_error():
    middleware = ModelCallTimeoutMiddleware(timeout_s=0.01)

    async def handler(request):
        await asyncio.sleep(1)

    try:
        asyncio.run(middleware.awrap_model_call(_request(), handler))
    except TimeoutError as exc:
        assert str(exc) == "model call exceeded 0.01s"
    else:
        raise AssertionError("Expected model call to time out")


def test_timeout_allows_model_fallback_to_run():
    primary_model = object()
    fallback_model = object()
    timeout_middleware = ModelCallTimeoutMiddleware(timeout_s=0.01)
    fallback_middleware = ModelFallbackMiddleware(fallback_model)
    response = ModelResponse([AIMessage(content="fallback")])
    calls = []

    async def handler(request):
        calls.append(request.model)
        if request.model is primary_model:
            await asyncio.sleep(1)
        return response

    async def composed_handler(request):
        return await timeout_middleware.awrap_model_call(request, handler)

    result = asyncio.run(
        fallback_middleware.awrap_model_call(_request(primary_model), composed_handler)
    )

    assert result is response
    assert calls == [primary_model, fallback_model]
