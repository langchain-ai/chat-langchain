import asyncio

from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage

from src.middleware.model_failure_boundary_middleware import (
    ModelFailureBoundaryMiddleware,
)


class AnthropicInvalidRequestError(Exception):
    status_code = 400


def test_provider_400_returns_last_non_empty_ai_draft():
    middleware = ModelFailureBoundaryMiddleware()
    request = ModelRequest(
        model=object(),
        messages=[
            HumanMessage(content="How do I authenticate?"),
            AIMessage(content="Here is a draft answer."),
        ],
    )

    async def handler(request: ModelRequest) -> ModelResponse:
        raise AnthropicInvalidRequestError("invalid_request_error")

    result = asyncio.run(middleware.awrap_model_call(request, handler))

    assert isinstance(result.result[0], AIMessage)
    assert result.result[0].content == "Here is a draft answer."


def test_provider_400_returns_failure_notice_without_draft():
    middleware = ModelFailureBoundaryMiddleware()
    request = ModelRequest(
        model=object(), messages=[HumanMessage(content="How do I authenticate?")]
    )

    async def handler(request: ModelRequest) -> ModelResponse:
        raise AnthropicInvalidRequestError("invalid_request_error")

    result = asyncio.run(middleware.awrap_model_call(request, handler))

    assert isinstance(result.result[0], AIMessage)
    assert result.result[0].content
