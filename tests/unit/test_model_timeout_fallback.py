"""Tests for model timeout and fallback behavior."""

import asyncio

from langchain.agents.middleware import (
    ModelFallbackMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from src.middleware.retry_middleware import (
    ModelRetryMiddleware,
    ModelTimeoutMiddleware,
)


class HangingChatModel(BaseChatModel):
    """Chat model that never completes asynchronous calls."""

    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "hanging"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ARG002
        return ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content="unexpected sync response"))
            ]
        )

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ARG002
        self.calls += 1
        await asyncio.Future()


class AnsweringChatModel(BaseChatModel):
    """Chat model that returns a successful response."""

    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "answering"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ARG002
        self.calls += 1
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="fallback answer"))]
        )

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ARG002
        self.calls += 1
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="fallback answer"))]
        )


def test_timeout_retries_primary_then_reaches_fallback():
    async def run_test():
        primary = HangingChatModel()
        fallback = AnsweringChatModel()
        request = ModelRequest(
            model=primary,
            messages=[HumanMessage(content="hello")],
        )
        timeout = ModelTimeoutMiddleware(timeout=0.01)
        retry = ModelRetryMiddleware(max_retries=1, initial_delay=0)
        fallbacks = ModelFallbackMiddleware(fallback)

        async def invoke(model_request):
            response = await model_request.model.ainvoke(model_request.messages)
            return ModelResponse(result=[response])

        async def retry_call(model_request):
            return await retry.awrap_model_call(
                model_request,
                lambda current_request: timeout.awrap_model_call(
                    current_request, invoke
                ),
            )

        result = await asyncio.wait_for(
            fallbacks.awrap_model_call(request, retry_call),
            timeout=0.2,
        )

        assert result.result[0].content == "fallback answer"
        assert primary.calls == 2
        assert fallback.calls == 1

    asyncio.run(run_test())
