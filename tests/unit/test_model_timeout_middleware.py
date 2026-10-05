import asyncio

from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage

from src.middleware.model_timeout_middleware import ModelCallTimeoutMiddleware


def test_model_call_timeout_raises_timeout_error():
    asyncio.run(_test_model_call_timeout_raises_timeout_error())


async def _test_model_call_timeout_raises_timeout_error():
    middleware = ModelCallTimeoutMiddleware(timeout_seconds=0.01)

    async def handler(request):  # noqa: ARG001
        await asyncio.sleep(1)
        return ModelResponse(result=[AIMessage(content="unreachable")])

    try:
        await middleware.awrap_model_call(
            ModelRequest(model=object(), messages=[]), handler
        )
    except TimeoutError as exc:
        assert str(exc) == "Model call exceeded 0.01s"
    else:
        raise AssertionError("Expected model call to time out")


def test_model_call_timeout_allows_fallback_model():
    asyncio.run(_test_model_call_timeout_allows_fallback_model())


async def _test_model_call_timeout_allows_fallback_model():
    class FakeModel:
        _llm_type = "fake-chat"

    primary = FakeModel()
    fallback = FakeModel()
    timeout_middleware = ModelCallTimeoutMiddleware(timeout_seconds=0.01)
    fallback_middleware = ModelFallbackMiddleware(fallback)
    invoked_models = []

    async def model_handler(request):
        invoked_models.append(request.model)
        if request.model is primary:
            await asyncio.sleep(1)
        return ModelResponse(result=[AIMessage(content="fallback response")])

    async def timeout_handler(request):
        return await timeout_middleware.awrap_model_call(request, model_handler)

    response = await fallback_middleware.awrap_model_call(
        ModelRequest(model=primary, messages=[]), timeout_handler
    )

    assert response.result[0].content == "fallback response"
    assert invoked_models == [primary, fallback]
