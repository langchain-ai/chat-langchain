import asyncio

import pytest

import src.middleware.retry_middleware as retry_module
from src.middleware.retry_middleware import ModelRetryMiddleware


def test_retry_middleware_retries_timeout_and_raises(monkeypatch):
    monkeypatch.setattr(retry_module, "MODEL_CALL_TIMEOUT_SECONDS", 0.01)
    middleware = ModelRetryMiddleware(max_retries=2, initial_delay=0)
    calls = 0

    async def handler(request):  # noqa: ARG001
        nonlocal calls
        calls += 1
        await asyncio.sleep(1)

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(middleware.awrap_model_call(None, handler))

    assert calls == 3


def test_retry_middleware_returns_fast_handler_result(monkeypatch):
    monkeypatch.setattr(retry_module, "MODEL_CALL_TIMEOUT_SECONDS", 0.01)
    middleware = ModelRetryMiddleware(max_retries=2, initial_delay=0)
    calls = 0

    async def handler(request):  # noqa: ARG001
        nonlocal calls
        calls += 1
        return "result"

    result = asyncio.run(middleware.awrap_model_call(None, handler))

    assert result == "result"
    assert calls == 1
