import asyncio

import pytest
from anthropic import AuthenticationError as AnthropicAuthenticationError
from anthropic import PermissionDeniedError as AnthropicPermissionDeniedError
from google.genai.errors import ClientError
from httpx import HTTPStatusError, Request, Response
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_google_genai.chat_models import GoogleInvalidRequestError
from openai import AuthenticationError as OpenAIAuthenticationError
from openai import PermissionDeniedError as OpenAIPermissionDeniedError

from src.middleware.retry_middleware import (
    ModelRetryMiddleware,
    _ProviderValidationAwareRunnableRetry,
    is_provider_auth_error,
)


def _http_response(status):
    return Response(status, request=Request("POST", "https://provider.example"))


@pytest.mark.parametrize(
    "error",
    [
        GoogleInvalidRequestError("400 INVALID_ARGUMENT: API key not valid"),
        GoogleInvalidRequestError("400: API_KEY_INVALID"),
        ClientError(
            400,
            {
                "error": {
                    "status": "INVALID_ARGUMENT",
                    "message": "API key not valid",
                }
            },
        ),
        ClientError(401, {"error": {"message": "Unauthorized"}}),
        ClientError(403, {"error": {"message": "Forbidden"}}),
        *[
            error_type("denied", response=_http_response(status), body=None)
            for error_type, status in [
                (OpenAIAuthenticationError, 401),
                (OpenAIPermissionDeniedError, 403),
                (AnthropicAuthenticationError, 401),
                (AnthropicPermissionDeniedError, 403),
            ]
        ],
        *[
            HTTPStatusError("denied", request=response.request, response=response)
            for response in [_http_response(401), _http_response(403)]
        ],
    ],
)
def test_provider_auth_errors(error):
    assert is_provider_auth_error(error)
    wrapped = RuntimeError("provider call failed")
    wrapped.__cause__ = error
    assert is_provider_auth_error(wrapped)
    wrapped.__cause__ = None
    wrapped.__context__ = error
    wrapped.__suppress_context__ = False
    assert is_provider_auth_error(wrapped)


@pytest.mark.parametrize(
    "error",
    [
        GoogleInvalidRequestError("400 INVALID_ARGUMENT: invalid tool schema"),
        ClientError(400, {"error": {"message": "Invalid tool schema"}}),
        RuntimeError("API key not valid"),
        ValueError("unsupported request shape"),
        TimeoutError("timed out"),
        *[
            HTTPStatusError("failed", request=response.request, response=response)
            for response in [
                _http_response(400),
                _http_response(429),
                _http_response(503),
            ]
        ],
    ],
)
def test_non_auth_errors(error):
    assert not is_provider_auth_error(error)


def test_classifier_handles_exception_cycles_and_suppressed_context():
    error = RuntimeError("not auth")
    error.__cause__ = error
    assert not is_provider_auth_error(error)
    error.__cause__ = None
    error.__context__ = GoogleInvalidRequestError("API_KEY_INVALID")
    error.__suppress_context__ = True
    assert not is_provider_auth_error(error)


@pytest.mark.parametrize("auth_failure", [True, False])
@pytest.mark.parametrize("use_async", [True, False])
def test_runnable_retry_policy(auth_failure, use_async):
    calls = 0
    error = (
        GoogleInvalidRequestError("400 INVALID_ARGUMENT: API key not valid")
        if auth_failure
        else TimeoutError("timed out")
    )

    def invoke(_input):
        nonlocal calls
        calls += 1
        raise error

    runnable = _ProviderValidationAwareRunnableRetry(
        bound=RunnableLambda(invoke),
        max_attempt_number=3,
        wait_exponential_jitter=False,
    )
    with pytest.raises(type(error)):
        if use_async:
            asyncio.run(runnable.ainvoke("request"))
        else:
            runnable.invoke("request")
    assert calls == (1 if auth_failure else 3)


@pytest.mark.parametrize("auth_failure", [True, False])
def test_middleware_retry_policy(auth_failure):
    middleware = ModelRetryMiddleware(max_retries=2, initial_delay=0)
    calls = 0
    error = (
        GoogleInvalidRequestError("API_KEY_INVALID")
        if auth_failure
        else TimeoutError("timed out")
    )

    async def handler(request):
        nonlocal calls
        calls += 1
        raise error

    with pytest.raises(type(error)):
        asyncio.run(
            middleware.awrap_model_call(
                ModelRequest(model=object(), messages=[HumanMessage(content="Hi")]),
                handler,
            )
        )
    assert calls == (1 if auth_failure else 3)


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
