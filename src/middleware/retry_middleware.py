"""Retry middleware for model calls and provider responses."""

import asyncio
import logging
import threading
from typing import Awaitable, Callable

from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.model_fallback import (
    _sanitize_request_for_fallback,
)
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.runnables.retry import RunnableRetry
from langgraph.config import get_config
from tenacity import retry_if_exception

logger = logging.getLogger(__name__)

# Finish reasons that indicate a retryable failure (not an exception)
RETRYABLE_FINISH_REASONS = {
    "MALFORMED_FUNCTION_CALL",  # Gemini: invalid tool call syntax
}


class MalformedResponseError(Exception):
    """Raised when model returns a malformed response after exhausting retries."""

    pass


class PrimaryModelCircuitBreaker:
    """Track whether the primary model has a permanent authentication failure."""

    def __init__(self) -> None:
        """Initialize a closed circuit breaker."""
        self._lock = threading.Lock()
        self.auth_error_count = 0
        self.open = False

    def trip(self) -> bool:
        """Open the breaker and report whether this is the first trip."""
        with self._lock:
            was_open = self.open
            self.auth_error_count += 1
            self.open = True
            return not was_open


PRIMARY_MODEL_CIRCUIT_BREAKER = PrimaryModelCircuitBreaker()


def _model_identifier(model: object) -> str:
    return str(
        getattr(model, "model_name", None)
        or getattr(model, "model", None)
        or getattr(model, "model_id", None)
        or ""
    )


def _is_authentication_error(error: Exception) -> bool:
    error_name = type(error).__name__
    if error_name in {"AuthenticationError", "GoogleInvalidRequestError"}:
        return True

    status_code = getattr(error, "status_code", None)
    response = getattr(error, "response", None)
    status_code = status_code or getattr(response, "status_code", None)
    if status_code in {401, 403}:
        return True

    message = str(error).upper()
    return "API_KEY_INVALID" in message or (
        "INVALID_ARGUMENT" in message and "KEY" in message
    )


def _mark_fallback_served() -> None:
    try:
        metadata = get_config().setdefault("metadata", {})
    except RuntimeError:
        return
    metadata.update(
        {
            "served_by_fallback": True,
            "primary_model_error": "auth",
        }
    )


class _ProviderValidationAwareRunnableRetry(RunnableRetry):
    @property
    def _kwargs_retrying(self) -> dict[str, object]:
        kwargs = super()._kwargs_retrying
        kwargs["retry"] = retry_if_exception(
            lambda exception: not isinstance(exception, ValueError)
        )
        return kwargs


class ModelRetryMiddleware(AgentMiddleware):
    """Retry transient model failures and malformed responses."""

    def __init__(
        self,
        max_retries: int = 2,
        initial_delay: float = 0.5,
        backoff_factor: float = 2.0,
        primary_model_id: str = "google_genai:gemini-3.5-flash-lite",
        circuit_breaker: PrimaryModelCircuitBreaker | None = None,
    ):
        """Configure retry attempts and backoff timing."""
        super().__init__()
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.primary_model_id = primary_model_id
        self.circuit_breaker = circuit_breaker or PRIMARY_MODEL_CIRCUIT_BREAKER

    def _get_finish_reason(self, response: ModelResponse) -> str:
        """Extract finish_reason from response metadata."""
        metadata = getattr(response, "response_metadata", None) or {}
        return metadata.get("finish_reason", "")

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Retry transient failures from the wrapped model handler."""
        last_exception: Exception | None = None
        last_retryable_reason: str | None = None
        primary_model_name = self.primary_model_id.split(":", 1)[-1]
        is_primary = _model_identifier(request.model) in {
            self.primary_model_id,
            primary_model_name,
        }

        if is_primary and self.circuit_breaker.open:
            raise RuntimeError("Primary model circuit breaker is open")

        for attempt in range(self.max_retries + 1):
            try:
                response = await handler(request)
                finish_reason = self._get_finish_reason(response)

                if finish_reason in RETRYABLE_FINISH_REASONS:
                    if attempt < self.max_retries:
                        delay = self.initial_delay * (self.backoff_factor**attempt)
                        logger.warning(
                            f"Retryable response ({finish_reason}) "
                            f"attempt {attempt + 1}/{self.max_retries + 1}, "
                            f"retrying in {delay:.2f}s"
                        )
                        last_retryable_reason = finish_reason
                        await asyncio.sleep(delay)
                        continue

                return response

            except Exception as e:
                if isinstance(e, ValueError):
                    raise
                if is_primary and _is_authentication_error(e):
                    first_failure = self.circuit_breaker.trip()
                    if first_failure:
                        logger.error(
                            "Primary model authentication failed for %s; opening circuit breaker",
                            self.primary_model_id,
                        )
                    raise
                last_exception = e
                if attempt < self.max_retries:
                    delay = self.initial_delay * (self.backoff_factor**attempt)
                    logger.warning(
                        f"Model call failed attempt {attempt + 1}/{self.max_retries + 1}: {e}, "
                        f"retrying in {delay:.2f}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Model call failed after {self.max_retries + 1} attempts: {e}"
                    )

        # Exhausted retries - raise for fallback middleware
        if last_exception:
            raise last_exception

        if last_retryable_reason:
            raise MalformedResponseError(
                f"Model returned {last_retryable_reason} after {self.max_retries + 1} attempts"
            )

        raise RuntimeError("Unexpected state in retry middleware")


class AuthAwareModelFallbackMiddleware(ModelFallbackMiddleware):
    """Skip an authenticated primary model after its circuit breaker opens."""

    def __init__(
        self,
        first_model: str,
        *additional_models: str,
        primary_model_id: str = "google_genai:gemini-3.5-flash-lite",
        circuit_breaker: PrimaryModelCircuitBreaker | None = None,
    ) -> None:
        """Initialize fallback models and the shared circuit breaker."""
        super().__init__(first_model, *additional_models)
        self.primary_model_id = primary_model_id
        self.circuit_breaker = circuit_breaker or PRIMARY_MODEL_CIRCUIT_BREAKER

    async def awrap_model_call(self, request, handler):
        """Route calls to fallback models when the primary circuit is open."""
        if self.circuit_breaker.open:
            _mark_fallback_served()
            return await self._call_fallbacks(request, handler)

        try:
            return await handler(request)
        except Exception:
            if self.circuit_breaker.open:
                _mark_fallback_served()
            return await self._call_fallbacks(request, handler)

    async def _call_fallbacks(self, request, handler):
        last_exception = None
        for fallback_model in self.models:
            fallback_request = _sanitize_request_for_fallback(request, fallback_model)
            try:
                return await handler(fallback_request.override(model=fallback_model))
            except Exception as error:
                last_exception = error
        if last_exception:
            raise last_exception
        raise RuntimeError("No fallback models configured")

    def wrap_model_call(self, request, handler):
        """Route synchronous calls to fallback models when needed."""
        if self.circuit_breaker.open:
            _mark_fallback_served()
            return self._call_fallbacks_sync(request, handler)

        try:
            return handler(request)
        except Exception:
            if self.circuit_breaker.open:
                _mark_fallback_served()
            return self._call_fallbacks_sync(request, handler)

    def _call_fallbacks_sync(self, request, handler):
        last_exception = None
        for fallback_model in self.models:
            fallback_request = _sanitize_request_for_fallback(request, fallback_model)
            try:
                return handler(fallback_request.override(model=fallback_model))
            except Exception as error:
                last_exception = error
        if last_exception:
            raise last_exception
        raise RuntimeError("No fallback models configured")


__all__ = [
    "AuthAwareModelFallbackMiddleware",
    "ModelRetryMiddleware",
    "MalformedResponseError",
    "PRIMARY_MODEL_CIRCUIT_BREAKER",
    "PrimaryModelCircuitBreaker",
]
