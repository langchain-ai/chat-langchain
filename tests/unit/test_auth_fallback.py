import asyncio

from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import HumanMessage

from src.middleware.retry_middleware import (
    AuthenticationAwareModelFallbackMiddleware,
    AuthenticationCircuitBreaker,
    is_authentication_error,
    mark_fallback_used,
)


class ProviderError(Exception):
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


def _request():
    return ModelRequest(model=object(), messages=[HumanMessage(content="Hi")])


def test_authentication_error_shapes_are_classified():
    assert is_authentication_error(ProviderError("API_KEY_INVALID"))
    assert is_authentication_error(ProviderError("PERMISSION_DENIED"))
    assert is_authentication_error(ProviderError("UNAUTHENTICATED"))
    assert is_authentication_error(ProviderError("unauthorized", 401))
    assert is_authentication_error(ProviderError("forbidden", 403))
    assert not is_authentication_error(ProviderError("temporary outage", 500))


def test_breaker_opens_and_closes_after_cooldown(monkeypatch):
    now = 100.0
    monkeypatch.setattr("src.middleware.retry_middleware.time.monotonic", lambda: now)
    breaker = AuthenticationCircuitBreaker("google:model", 10)
    breaker.open(ProviderError("API_KEY_INVALID"))
    assert breaker.is_open
    now = 111.0
    assert not breaker.is_open


def test_fallback_skips_primary_while_breaker_is_open(monkeypatch):
    middleware = AuthenticationAwareModelFallbackMiddleware(
        "google:model", "openai:model", cooldown_seconds=60
    )
    calls = []

    async def handler(request):
        calls.append(request.model)
        if len(calls) == 1:
            raise ProviderError("API_KEY_INVALID")
        return ModelResponse(result=[])

    asyncio.run(middleware.awrap_model_call(_request(), handler))
    asyncio.run(middleware.awrap_model_call(_request(), handler))
    assert len(calls) == 3
    assert calls[0] is not calls[1]
    assert calls[1] is calls[2]


def test_fallback_metadata_records_serving_model(monkeypatch):
    metadata = {}

    class RunTree:
        pass

    run_tree = RunTree()
    run_tree.metadata = metadata
    monkeypatch.setattr(
        "src.middleware.retry_middleware.get_current_run_tree", lambda: run_tree
    )

    mark_fallback_used("openai:gpt-5.4-nano")

    assert metadata == {
        "fallback_used": True,
        "served_by_model": "openai:gpt-5.4-nano",
    }
