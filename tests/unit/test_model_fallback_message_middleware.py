"""Tests for the terminal model-error fallback middleware."""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage

from src.middleware.model_fallback_message_middleware import (
    DEFAULT_FALLBACK_MESSAGE,
    ModelErrorFallbackMiddleware,
)


class _OpenAIAPIError(Exception):
    pass


async def _ok_handler(request):
    return AIMessage(content="real answer")


def test_returns_response_when_model_succeeds():
    middleware = ModelErrorFallbackMiddleware()

    result = asyncio.run(middleware.awrap_model_call("req", _ok_handler))

    assert result.content == "real answer"


def test_transient_openai_apierror_becomes_user_message():
    middleware = ModelErrorFallbackMiddleware()

    async def handler(request):
        raise _OpenAIAPIError(
            "The server had an error. See help.openai.com. (request ID req_abc123)"
        )

    result = asyncio.run(middleware.awrap_model_call("req", handler))

    assert isinstance(result, AIMessage)
    assert result.content == DEFAULT_FALLBACK_MESSAGE


def test_transient_500_status_code_becomes_user_message():
    middleware = ModelErrorFallbackMiddleware()

    class _ServerError(Exception):
        status_code = 500

    async def handler(request):
        raise _ServerError("boom")

    result = asyncio.run(middleware.awrap_model_call("req", handler))

    assert isinstance(result, AIMessage)
    assert result.content == DEFAULT_FALLBACK_MESSAGE


def test_non_transient_error_is_reraised():
    middleware = ModelErrorFallbackMiddleware()

    async def handler(request):
        raise ValueError("invalid request payload")

    with pytest.raises(ValueError):
        asyncio.run(middleware.awrap_model_call("req", handler))
