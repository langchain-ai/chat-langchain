import asyncio

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from src.middleware import retry_middleware
from src.middleware.retry_middleware import (
    ModelRetryMiddleware,
    ProviderCircuitOpenError,
    _ProviderValidationAwareRunnableRetry,
    is_auth_error,
    reset_provider_circuits,
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

@pytest.mark.parametrize(
    "exception",
    [
        RuntimeError("400 INVALID_ARGUMENT API_KEY_INVALID"),
        type("HttpError", (Exception,), {"status_code": 401})("unauthorized"),
        type("HttpError", (Exception,), {"response": type("Response", (), {"status_code": 403})()})(),
        type("Unauthenticated", (Exception,), {})(),
        type("PermissionDenied", (Exception,), {})(),
        type("AuthenticationError", (Exception,), {})(),
    ],
)
def test_auth_error_classifier_recognizes_provider_credential_failures(exception):
    assert is_auth_error(exception)


@pytest.mark.parametrize(
    "exception",
    [
        RuntimeError("400 INVALID_ARGUMENT invalid tool schema"),
        type("HttpError", (Exception,), {"status_code": 400})("bad request"),
        RuntimeError("temporary upstream timeout"),
    ],
)
def test_auth_error_classifier_ignores_non_credential_failures(exception):
    assert not is_auth_error(exception)


def test_auth_failure_opens_breaker_skips_provider_and_closes_after_cooldown():
    reset_provider_circuits()
    middleware = ModelRetryMiddleware(max_retries=2, initial_delay=0)
    model = type("Model", (), {"model_name": "google_genai:gemini-test"})()
    request = ModelRequest(model=model, messages=[HumanMessage(content="Hi")])
    calls = 0

    async def auth_failure(_request):
        nonlocal calls
        calls += 1
        raise RuntimeError("API_KEY_INVALID")

    with pytest.raises(RuntimeError, match="API_KEY_INVALID"):
        asyncio.run(middleware.awrap_model_call(request, auth_failure))
    assert calls == 1

    with pytest.raises(ProviderCircuitOpenError):
        asyncio.run(middleware.awrap_model_call(request, auth_failure))
    assert calls == 1

    retry_middleware._auth_failure_until["google_genai:gemini-test"] = 0

    async def success(_request):
        return ModelResponse(
            result=[
                AIMessage(
                    content="answer",
                    response_metadata={"model_name": "gpt-5.4-nano"},
                )
            ]
        )

    response = asyncio.run(middleware.awrap_model_call(request, success))
    assert response.result[0].response_metadata["model_name"] == "gpt-5.4-nano"
    assert "google_genai:gemini-test" not in retry_middleware._auth_failure_until


def test_served_model_records_fallback_response_on_root_trace(monkeypatch):
    reset_provider_circuits()
    root_run = type("Run", (), {"metadata": {}, "parent_run": None})()
    child_run = type("Run", (), {"metadata": {}, "parent_run": root_run})()
    monkeypatch.setattr(retry_middleware, "get_current_run_tree", lambda: child_run)
    middleware = ModelRetryMiddleware(max_retries=0, initial_delay=0)
    primary = type("Model", (), {"model_name": "google_genai:gemini-test"})()
    request = ModelRequest(model=primary, messages=[HumanMessage(content="Hi")])

    async def fallback(_request):
        return ModelResponse(
            result=[
                AIMessage(
                    content="answer",
                    response_metadata={"model_name": "gpt-5.4-nano"},
                )
            ]
        )

    asyncio.run(middleware.awrap_model_call(request, fallback))
    assert root_run.metadata["served_model"] == "gpt-5.4-nano"
