import asyncio
from types import SimpleNamespace

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from src.middleware.retry_middleware import (
    ModelRetryMiddleware,
    _ProviderValidationAwareRunnableRetry,
)


class AnthropicBadRequestError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.response = SimpleNamespace(status_code=400)


class OpenAIInvalidRequestError(Exception):
    status_code = 400


class ProviderHTTPError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


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


def test_anthropic_tool_choice_bad_request_is_not_retried():
    middleware = ModelRetryMiddleware(max_retries=2, initial_delay=0)
    calls = 0

    async def handler(request: ModelRequest):
        nonlocal calls
        calls += 1
        raise AnthropicBadRequestError("tool_choice is invalid")

    with pytest.raises(AnthropicBadRequestError, match="tool_choice is invalid"):
        asyncio.run(
            middleware.awrap_model_call(
                ModelRequest(model=object(), messages=[HumanMessage(content="Hi")]),
                handler,
            )
        )

    assert calls == 1


def test_openai_function_name_bad_request_is_not_retried():
    middleware = ModelRetryMiddleware(max_retries=2, initial_delay=0)
    calls = 0

    async def handler(request: ModelRequest):
        nonlocal calls
        calls += 1
        raise OpenAIInvalidRequestError("invalid function.name")

    with pytest.raises(OpenAIInvalidRequestError, match="invalid function.name"):
        asyncio.run(
            middleware.awrap_model_call(
                ModelRequest(model=object(), messages=[HumanMessage(content="Hi")]),
                handler,
            )
        )

    assert calls == 1


@pytest.mark.parametrize("status_code", [429, 503])
def test_transient_http_errors_are_retried(status_code):
    middleware = ModelRetryMiddleware(max_retries=2, initial_delay=0)
    calls = 0

    async def handler(request: ModelRequest):
        nonlocal calls
        calls += 1
        raise ProviderHTTPError(status_code)

    with pytest.raises(ProviderHTTPError, match=f"HTTP {status_code}"):
        asyncio.run(
            middleware.awrap_model_call(
                ModelRequest(model=object(), messages=[HumanMessage(content="Hi")]),
                handler,
            )
        )

    assert calls == 3


def test_malformed_function_call_response_is_retried():
    middleware = ModelRetryMiddleware(max_retries=2, initial_delay=0)
    calls = 0

    async def handler(request: ModelRequest):
        nonlocal calls
        calls += 1
        response = ModelResponse(result=[AIMessage(content="malformed")])
        response.response_metadata = {"finish_reason": "MALFORMED_FUNCTION_CALL"}
        return response

    result = asyncio.run(
        middleware.awrap_model_call(
            ModelRequest(model=object(), messages=[HumanMessage(content="Hi")]),
            handler,
        )
    )

    assert calls == 3
    assert result.response_metadata["finish_reason"] == "MALFORMED_FUNCTION_CALL"
