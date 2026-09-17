"""Retry middleware for model calls and provider responses."""

import asyncio
import logging
from typing import Awaitable, Callable

from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.model_fallback import (
    _sanitize_request_for_fallback,
    _supports_anthropic_cache_control,
)
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.runnables.retry import RunnableRetry
from langgraph.errors import GraphBubbleUp
from tenacity import retry_if_exception

logger = logging.getLogger(__name__)

# Finish reasons that indicate a retryable failure (not an exception)
RETRYABLE_FINISH_REASONS = {
    "MALFORMED_FUNCTION_CALL",  # Gemini: invalid tool call syntax
}

_logged_permanent_request_errors: set[int] = set()


def _is_permanent_request_error(exc: BaseException) -> bool:
    """Return whether an exception indicates a permanently invalid request."""
    response = getattr(exc, "response", None)
    is_permanent = (
        isinstance(exc, ValueError)
        or getattr(exc, "status_code", None) == 400
        or getattr(response, "status_code", None) == 400
        or type(exc).__name__.endswith("BadRequestError")
        or "invalid_request_error" in str(exc).lower()
    )
    if is_permanent and id(exc) not in _logged_permanent_request_errors:
        _logged_permanent_request_errors.add(id(exc))
        provider = (
            getattr(exc, "provider", None) or type(exc).__module__.split(".", 1)[0]
        )
        logger.error("Permanent %s request error: %s", provider, exc)
    return is_permanent


class MalformedResponseError(Exception):
    """Raised when model returns a malformed response after exhausting retries."""

    pass


class _ProviderValidationAwareRunnableRetry(RunnableRetry):
    @property
    def _kwargs_retrying(self) -> dict[str, object]:
        kwargs = super()._kwargs_retrying
        kwargs["retry"] = retry_if_exception(
            lambda exception: not _is_permanent_request_error(exception)
        )
        return kwargs


class PermanentRequestAwareModelFallbackMiddleware(ModelFallbackMiddleware):
    """Skip fallback models for permanently invalid requests."""

    def wrap_model_call(self, request, handler):
        """Try fallback models only for non-permanent failures."""
        try:
            return handler(request)
        except GraphBubbleUp:
            raise
        except Exception as exc:
            if _is_permanent_request_error(exc):
                raise
            last_exception = exc

        for fallback_model in self.models:
            fallback_request = (
                request
                if _supports_anthropic_cache_control(fallback_model)
                else _sanitize_request_for_fallback(request)
            )
            try:
                return handler(fallback_request.override(model=fallback_model))
            except GraphBubbleUp:
                raise
            except Exception as exc:
                if _is_permanent_request_error(exc):
                    raise
                last_exception = exc

        raise last_exception

    async def awrap_model_call(self, request, handler):
        """Try fallback models only for non-permanent failures asynchronously."""
        try:
            return await handler(request)
        except GraphBubbleUp:
            raise
        except Exception as exc:
            if _is_permanent_request_error(exc):
                raise
            last_exception = exc

        for fallback_model in self.models:
            fallback_request = (
                request
                if _supports_anthropic_cache_control(fallback_model)
                else _sanitize_request_for_fallback(request)
            )
            try:
                return await handler(fallback_request.override(model=fallback_model))
            except GraphBubbleUp:
                raise
            except Exception as exc:
                if _is_permanent_request_error(exc):
                    raise
                last_exception = exc

        raise last_exception


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
                if _is_permanent_request_error(e):
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


__all__ = [
    "ModelRetryMiddleware",
    "MalformedResponseError",
    "PermanentRequestAwareModelFallbackMiddleware",
]
