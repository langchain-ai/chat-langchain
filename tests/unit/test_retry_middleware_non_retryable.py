"""Tests for non-retryable validation errors in ModelRetryMiddleware."""

import pytest

from src.middleware.retry_middleware import ModelRetryMiddleware


@pytest.mark.asyncio
async def test_non_retryable_validation_error_fails_fast(monkeypatch):
    """A non-retryable provider validation error should raise after exactly one attempt."""
    sleeps: list[float] = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr("asyncio.sleep", fake_sleep)

    middleware = ModelRetryMiddleware(max_retries=2, initial_delay=0.01)

    attempts = {"count": 0}

    async def handler(_request):
        attempts["count"] += 1
        raise ValueError("Malformed url parameter: image_url is not a valid URL")

    with pytest.raises(ValueError, match="Malformed url parameter"):
        await middleware.awrap_model_call(request=object(), handler=handler)

    assert attempts["count"] == 1
    assert sleeps == []


@pytest.mark.asyncio
async def test_transient_error_still_retries(monkeypatch):
    """A non-validation exception should still go through the retry loop."""
    sleeps: list[float] = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr("asyncio.sleep", fake_sleep)

    middleware = ModelRetryMiddleware(max_retries=2, initial_delay=0.01)

    attempts = {"count": 0}

    async def handler(_request):
        attempts["count"] += 1
        raise RuntimeError("connection reset")

    with pytest.raises(RuntimeError, match="connection reset"):
        await middleware.awrap_model_call(request=object(), handler=handler)

    assert attempts["count"] == 3
    assert len(sleeps) == 2
