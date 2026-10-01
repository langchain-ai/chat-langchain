"""Retry middleware for model calls and provider responses."""

import asyncio
import logging
import os
import time
from typing import Awaitable, Callable

from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.runnables.retry import RunnableRetry
from langgraph.errors import GraphBubbleUp
from langsmith import get_current_run_tree
from tenacity import retry_if_exception

logger = logging.getLogger(__name__)

# Finish reasons that indicate a retryable failure (not an exception)
RETRYABLE_FINISH_REASONS = {
    "MALFORMED_FUNCTION_CALL",  # Gemini: invalid tool call syntax
}

AUTH_BREAKER_COOLDOWN_SECONDS = float(
    os.getenv("MODEL_AUTH_BREAKER_COOLDOWN_SECONDS", "300")
)


def is_authentication_error(exception: BaseException) -> bool:
    """Return whether an exception indicates invalid provider credentials."""
    status_code = getattr(exception, "status_code", None)
    response = getattr(exception, "response", None)
    response_status = getattr(response, "status_code", None)
    if status_code in {401, 403} or response_status in {401, 403}:
        return True

    text = str(exception).upper()
    return any(
        marker in text
        for marker in (
            "API_KEY_INVALID",
            "PERMISSION_DENIED",
            "UNAUTHENTICATED",
            "INVALID API KEY",
            "INVALID_API_KEY",
            "HTTP 401",
            "HTTP 403",
            "ERROR CODE: 401",
            "ERROR CODE: 403",
            "CODE: 401",
            "CODE: 403",
            "STATUS CODE: 401",
            "STATUS CODE: 403",
        )
    )


class AuthenticationCircuitBreaker:
    """Skip a provider after an authentication failure for a cool-down."""

    def __init__(self, model_id: str, cooldown_seconds: float):
        """Initialize a breaker for one provider model."""
        self.model_id = model_id
        self.cooldown_seconds = cooldown_seconds
        self._opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        """Return whether the provider is currently blocked."""
        if self._opened_at is None:
            return False
        if time.monotonic() - self._opened_at >= self.cooldown_seconds:
            self._opened_at = None
            return False
        return True

    def open(self, exception: BaseException) -> None:
        """Open the breaker and log only its first failure."""
        if not self.is_open:
            self._opened_at = time.monotonic()
            logger.error(
                "Authentication circuit opened for %s for %.0f seconds: %s",
                self.model_id,
                self.cooldown_seconds,
                exception,
            )


def mark_fallback_used(model_id: str) -> None:
    """Attach fallback usage to the active LangSmith run."""
    try:
        run_tree = get_current_run_tree()
        if run_tree:
            run_tree.metadata["fallback_used"] = True
            run_tree.metadata["served_by_model"] = model_id
    except Exception:
        logger.debug("Unable to attach fallback metadata", exc_info=True)


class MalformedResponseError(Exception):
    """Raised when model returns a malformed response after exhausting retries."""

    pass


class _ProviderValidationAwareRunnableRetry(RunnableRetry):
    @property
    def _kwargs_retrying(self) -> dict[str, object]:
        kwargs = super()._kwargs_retrying
        kwargs["retry"] = retry_if_exception(
            lambda exception: (
                not isinstance(exception, ValueError)
                and not is_authentication_error(exception)
            )
        )
        return kwargs


class ModelRetryMiddleware(AgentMiddleware):
    """Retry transient model failures and malformed responses."""

    def __init__(
        self,
        max_retries: int = 2,
        initial_delay: float = 0.5,
        backoff_factor: float = 2.0,
    ):
        """Configure retry attempts and backoff timing."""
        super().__init__()
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor

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
                if isinstance(e, ValueError) or is_authentication_error(e):
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


class AuthenticationAwareModelFallbackMiddleware(ModelFallbackMiddleware):
    """Fallback middleware with an authentication circuit breaker."""

    def __init__(
        self,
        primary_model_id: str,
        *fallback_models: str,
        cooldown_seconds: float = AUTH_BREAKER_COOLDOWN_SECONDS,
    ) -> None:
        """Initialize fallback models and their authentication breaker."""
        super().__init__(*fallback_models)
        self.primary_model_id = primary_model_id
        self.fallback_model_ids = list(fallback_models)
        self.breaker = AuthenticationCircuitBreaker(primary_model_id, cooldown_seconds)

    def _fallback_request(self, request, fallback_model):
        from langchain.agents.middleware.model_fallback import (
            _sanitize_request_for_fallback,
        )

        return _sanitize_request_for_fallback(request, fallback_model).override(
            model=fallback_model
        )

    def wrap_model_call(self, request, handler):
        """Call the primary unless its authentication breaker is open."""
        if self.breaker.is_open:
            return self._call_fallbacks(request, handler)
        try:
            return handler(request)
        except GraphBubbleUp:
            raise
        except Exception as exception:
            if is_authentication_error(exception):
                self.breaker.open(exception)
            return self._call_fallbacks(request, handler)

    def _call_fallbacks(self, request, handler):
        last_exception: Exception | None = None
        for fallback_model, fallback_model_id in zip(
            self.models, self.fallback_model_ids
        ):
            try:
                result = handler(self._fallback_request(request, fallback_model))
                mark_fallback_used(fallback_model_id)
                return result
            except GraphBubbleUp:
                raise
            except Exception as exception:
                last_exception = exception
        if last_exception:
            raise last_exception
        raise RuntimeError("No fallback models configured")

    async def awrap_model_call(self, request, handler):
        """Call the primary asynchronously unless its breaker is open."""
        if self.breaker.is_open:
            return await self._acall_fallbacks(request, handler)
        try:
            return await handler(request)
        except GraphBubbleUp:
            raise
        except Exception as exception:
            if is_authentication_error(exception):
                self.breaker.open(exception)
            return await self._acall_fallbacks(request, handler)

    async def _acall_fallbacks(self, request, handler):
        last_exception: Exception | None = None
        for fallback_model, fallback_model_id in zip(
            self.models, self.fallback_model_ids
        ):
            try:
                result = await handler(self._fallback_request(request, fallback_model))
                mark_fallback_used(fallback_model_id)
                return result
            except GraphBubbleUp:
                raise
            except Exception as exception:
                last_exception = exception
        if last_exception:
            raise last_exception
        raise RuntimeError("No fallback models configured")


__all__ = [
    "AUTH_BREAKER_COOLDOWN_SECONDS",
    "AuthenticationAwareModelFallbackMiddleware",
    "AuthenticationCircuitBreaker",
    "ModelRetryMiddleware",
    "MalformedResponseError",
    "is_authentication_error",
    "mark_fallback_used",
]
