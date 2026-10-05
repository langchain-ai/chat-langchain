import asyncio

import pytest
from langchain.agents.middleware import (
    ModelFallbackMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from src.middleware.model_timeout_middleware import ModelCallTimeoutMiddleware


class FakeChatModel(BaseChatModel):
    name: str

    def __init__(self, name: str) -> None:
        super().__init__(name=name)

    @property
    def _llm_type(self) -> str:
        return "fake-chat"

    @property
    def _identifying_params(self) -> dict[str, str]:
        return {"name": self.name}

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self.name))]
        )


def _request(model: BaseChatModel) -> ModelRequest:
    return ModelRequest(model=model, messages=[])


def test_timeout_middleware_raises_for_stalled_handler():
    middleware = ModelCallTimeoutMiddleware(timeout_s=0.01)

    async def handler(request):
        await asyncio.sleep(1)

    with pytest.raises(TimeoutError, match="FakeChatModel.*0.01 seconds"):
        asyncio.run(
            middleware.awrap_model_call(_request(FakeChatModel("primary")), handler)
        )


def test_timeout_middleware_allows_fallback_after_stalled_primary():
    primary = FakeChatModel("primary")
    fallback = FakeChatModel("fallback")
    timeout_middleware = ModelCallTimeoutMiddleware(timeout_s=0.01)
    fallback_middleware = ModelFallbackMiddleware(fallback)
    calls = []

    async def handler(request):
        calls.append(request.model.name)
        if request.model is primary:
            await asyncio.sleep(1)
        return ModelResponse(result=[AIMessage(content=request.model.name)])

    async def run():
        return await fallback_middleware.awrap_model_call(
            _request(primary),
            lambda request: timeout_middleware.awrap_model_call(request, handler),
        )

    response = asyncio.run(run())

    assert response.result[0].content == "fallback"
    assert calls == ["primary", "fallback"]


def test_timeout_middleware_passes_through_fast_handler():
    middleware = ModelCallTimeoutMiddleware(timeout_s=1)
    response = ModelResponse(result=[AIMessage(content="ok")])

    async def handler(request):
        return response

    assert (
        asyncio.run(
            middleware.awrap_model_call(_request(FakeChatModel("primary")), handler)
        )
        is response
    )
