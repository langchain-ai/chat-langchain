"""Unit tests for the provider-agnostic ModelRetryMiddleware.

Covers issue b3db1839 (Anthropic 500 InternalServerError crashes deep_agent
with no retry or fallback) and the related provider-error classes:
- Anthropic InternalServerError (500): retried 3x then user-facing fallback
- Anthropic 400: NOT retried (deterministic client error)
- Anthropic OverloadedError (529): retried
- Google ChatGoogleGenerativeAIError 429 / Gemini rate limit: retried
- Cron-triggered runs: structured failure event logged
"""
import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage

from src.middleware.retry_middleware import (
    USER_FACING_FALLBACK_MESSAGE,
    ModelRetryMiddleware,
    _is_transient_provider_error,
)


# ---------------------------------------------------------------------------
# Fake provider exceptions
# ---------------------------------------------------------------------------


class InternalServerError(Exception):
    """Mirrors anthropic.InternalServerError (status_code 500)."""

    def __init__(self, message: str = "Anthropic 500", status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code


class OverloadedError(Exception):
    """Mirrors anthropic.OverloadedError (status_code 529)."""

    def __init__(self, message: str = "Overloaded"):
        super().__init__(message)
        self.status_code = 529


class APIStatusError(Exception):
    """Mirrors openai.APIStatusError -- carries an explicit status code."""

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


class ChatGoogleGenerativeAIError(Exception):
    """Mirrors langchain_google_genai.ChatGoogleGenerativeAIError."""

    def __init__(self, message: str = "Gemini 429: rate limit"):
        super().__init__(message)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(metadata: dict | None = None) -> SimpleNamespace:
    """Build a minimal ModelRequest-like object for awrap_model_call."""
    config = {}
    if metadata is not None:
        config["metadata"] = metadata
    runtime = SimpleNamespace(config=config)
    return SimpleNamespace(runtime=runtime, model=None, messages=[])


# ---------------------------------------------------------------------------
# Classification tests
# ---------------------------------------------------------------------------


class TestErrorClassification:
    def test_anthropic_500_is_transient(self):
        assert _is_transient_provider_error(InternalServerError()) is True

    def test_anthropic_400_is_not_transient(self):
        # 4xx other than 429 are deterministic client errors.
        err = APIStatusError("Bad request", status_code=400)
        assert _is_transient_provider_error(err) is False

    def test_anthropic_401_is_not_transient(self):
        err = APIStatusError("Unauthorized", status_code=401)
        assert _is_transient_provider_error(err) is False

    def test_429_is_transient(self):
        err = APIStatusError("Too many requests", status_code=429)
        assert _is_transient_provider_error(err) is True

    def test_overloaded_529_is_transient(self):
        assert _is_transient_provider_error(OverloadedError()) is True

    def test_gemini_error_is_transient_by_classname(self):
        assert _is_transient_provider_error(ChatGoogleGenerativeAIError()) is True

    def test_503_is_transient(self):
        err = APIStatusError("Service unavailable", status_code=503)
        assert _is_transient_provider_error(err) is True

    def test_message_substring_fallback(self):
        # No status_code, but message mentions internal server error.
        assert (
            _is_transient_provider_error(Exception("Internal Server Error"))
            is True
        )

    def test_generic_value_error_is_not_transient(self):
        assert _is_transient_provider_error(ValueError("bad input")) is False


# ---------------------------------------------------------------------------
# Retry behavior tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_500_retried_then_user_facing_fallback():
    """Anthropic 500 retried max_retries+1 times then fallback message emitted."""
    middleware = ModelRetryMiddleware(
        max_retries=2, initial_delay=0.0, jitter=0.0
    )
    handler = AsyncMock(side_effect=InternalServerError())
    request = _make_request()

    result = await middleware.awrap_model_call(request, handler)

    # 3 total attempts (max_retries=2 + initial)
    assert handler.await_count == 3
    # User-facing fallback message returned as a state update
    assert isinstance(result, dict)
    assert "messages" in result
    assert len(result["messages"]) == 1
    msg = result["messages"][0]
    assert isinstance(msg, AIMessage)
    assert msg.content == USER_FACING_FALLBACK_MESSAGE


@pytest.mark.asyncio
async def test_anthropic_400_not_retried():
    """Anthropic 400 (deterministic) raises immediately without retrying."""
    middleware = ModelRetryMiddleware(
        max_retries=2, initial_delay=0.0, jitter=0.0
    )
    err = APIStatusError("Bad request", status_code=400)
    handler = AsyncMock(side_effect=err)
    request = _make_request()

    with pytest.raises(APIStatusError):
        await middleware.awrap_model_call(request, handler)

    # Called exactly once -- no retry
    assert handler.await_count == 1


@pytest.mark.asyncio
async def test_overloaded_529_retried():
    """Anthropic OverloadedError (529) is retried like a 5xx."""
    middleware = ModelRetryMiddleware(
        max_retries=2, initial_delay=0.0, jitter=0.0
    )
    handler = AsyncMock(side_effect=OverloadedError())
    request = _make_request()

    result = await middleware.awrap_model_call(request, handler)

    assert handler.await_count == 3
    assert isinstance(result, dict)
    assert isinstance(result["messages"][0], AIMessage)


@pytest.mark.asyncio
async def test_gemini_429_retried():
    """ChatGoogleGenerativeAIError (Gemini 429) is retried."""
    middleware = ModelRetryMiddleware(
        max_retries=2, initial_delay=0.0, jitter=0.0
    )
    handler = AsyncMock(side_effect=ChatGoogleGenerativeAIError())
    request = _make_request()

    result = await middleware.awrap_model_call(request, handler)

    assert handler.await_count == 3
    assert isinstance(result, dict)
    assert isinstance(result["messages"][0], AIMessage)


@pytest.mark.asyncio
async def test_success_on_second_attempt_returns_response():
    """If a retry succeeds, the original response is returned (no fallback)."""

    class FakeResponse:
        response_metadata = {"finish_reason": "stop"}
        content = "ok"

    middleware = ModelRetryMiddleware(
        max_retries=2, initial_delay=0.0, jitter=0.0
    )
    handler = AsyncMock(
        side_effect=[InternalServerError(), FakeResponse()]
    )
    request = _make_request()

    result = await middleware.awrap_model_call(request, handler)

    assert handler.await_count == 2
    # Not a dict -- the real ModelResponse-like object is returned unchanged.
    assert not isinstance(result, dict)
    assert getattr(result, "content", None) == "ok"


@pytest.mark.asyncio
async def test_cron_triggered_failure_logs_structured_event(caplog):
    """For cron-triggered runs, a structured failure event is logged."""
    middleware = ModelRetryMiddleware(
        max_retries=1, initial_delay=0.0, jitter=0.0
    )
    handler = AsyncMock(side_effect=InternalServerError())
    request = _make_request(
        metadata={"source": "trigger", "cron_id": "cron-abc", "thread_id": "t-1"}
    )

    with caplog.at_level(logging.ERROR, logger="src.middleware.retry_middleware"):
        result = await middleware.awrap_model_call(request, handler)

    # Fallback message still returned
    assert isinstance(result, dict)

    # Structured cron-failure record was logged
    cron_records = [
        r
        for r in caplog.records
        if getattr(r, "structured", {}).get("event") == "cron_model_failure"
    ]
    assert cron_records, (
        "Expected a structured cron_model_failure log; got: "
        f"{[r.getMessage() for r in caplog.records]}"
    )
    structured = cron_records[0].structured
    assert structured["cron_id"] == "cron-abc"
    assert structured["source"] == "trigger"
    assert structured["error_type"] == "InternalServerError"
    assert structured["attempts"] == 2


@pytest.mark.asyncio
async def test_non_cron_failure_does_not_log_cron_event(caplog):
    """A regular interactive run failure does NOT emit the cron event."""
    middleware = ModelRetryMiddleware(
        max_retries=1, initial_delay=0.0, jitter=0.0
    )
    handler = AsyncMock(side_effect=InternalServerError())
    request = _make_request(metadata={"source": "ui"})

    with caplog.at_level(logging.ERROR, logger="src.middleware.retry_middleware"):
        result = await middleware.awrap_model_call(request, handler)

    assert isinstance(result, dict)
    cron_records = [
        r
        for r in caplog.records
        if getattr(r, "structured", {}).get("event") == "cron_model_failure"
    ]
    assert not cron_records
