"""Tests for ModelRetryMiddleware 4xx short-circuit behaviour.

Bug: ModelRetryMiddleware.awrap_model_call was retrying ALL exceptions,
including HTTP 4xx client errors (e.g. ChatXAI BadRequestError for images
with invalid dimensions). These requests can never succeed on retry, so
the retries only waste time (80-210 s per affected trace) before the
fallback middleware eventually takes over.

Fix: exceptions whose ``status_code`` attribute falls in [400, 500) are
re-raised immediately without any retry.
"""

import pytest

from src.middleware.retry_middleware import ModelRetryMiddleware


# ---------------------------------------------------------------------------
# Minimal fake exception class that mimics openai.BadRequestError
# ---------------------------------------------------------------------------


class _FakeAPIError(Exception):
    """Minimal stand-in for openai.APIStatusError / BadRequestError."""

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_middleware(max_retries: int = 2, initial_delay: float = 0.0) -> ModelRetryMiddleware:
    return ModelRetryMiddleware(
        max_retries=max_retries,
        initial_delay=initial_delay,
        backoff_factor=1.0,
    )


def _counting_handler(exc: Exception):
    """Return an async handler that always raises *exc* and counts calls."""
    calls = {"n": 0}

    async def handler(request):
        calls["n"] += 1
        raise exc

    return handler, calls


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_400_error_is_not_retried():
    """A 4xx error must be raised immediately; the handler is called exactly once."""
    middleware = _make_middleware(max_retries=2, initial_delay=0.0)
    exc = _FakeAPIError("Image dimensions 512x6 are too small", status_code=400)
    handler, calls = _counting_handler(exc)

    with pytest.raises(_FakeAPIError) as exc_info:
        await middleware.awrap_model_call(request=object(), handler=handler)

    assert calls["n"] == 1, "Handler should be called only once for a 4xx error"
    assert exc_info.value is exc


@pytest.mark.asyncio
async def test_500_error_is_retried():
    """A 5xx error must be retried up to max_retries times."""
    max_retries = 2
    middleware = _make_middleware(max_retries=max_retries, initial_delay=0.0)
    exc = _FakeAPIError("Internal server error", status_code=500)
    handler, calls = _counting_handler(exc)

    with pytest.raises(_FakeAPIError):
        await middleware.awrap_model_call(request=object(), handler=handler)

    # Expects 1 initial attempt + max_retries retries
    assert calls["n"] == max_retries + 1, (
        f"Handler should be called {max_retries + 1} times for a 5xx error, "
        f"got {calls['n']}"
    )


@pytest.mark.asyncio
async def test_exception_without_status_code_is_retried():
    """Generic exceptions with no status_code attribute must still be retried."""
    max_retries = 2
    middleware = _make_middleware(max_retries=max_retries, initial_delay=0.0)
    exc = RuntimeError("network blip")
    handler, calls = _counting_handler(exc)

    with pytest.raises(RuntimeError):
        await middleware.awrap_model_call(request=object(), handler=handler)

    assert calls["n"] == max_retries + 1
