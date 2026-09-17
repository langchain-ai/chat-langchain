"""Retry middleware for model calls and provider responses."""

import asyncio
import importlib
import logging
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
from tenacity import retry_if_exception

logger = logging.getLogger(__name__)

# Finish reasons that indicate a retryable failure (not an exception)
RETRYABLE_FINISH_REASONS = {
    "MALFORMED_FUNCTION_CALL",  # Gemini: invalid tool call syntax
}


class MalformedResponseError(Exception):
    """Raised when model returns a malformed response after exhausting retries."""

    pass


def is_provider_validation_error(exc: Exception) -> bool:
    """Return whether an exception represents a deterministic provider rejection."""
    if isinstance(exc, ValueError):
        return True

    exception_type = type(exc)
    if exception_type.__module__.split(".", 1)[0] in {
        "openai",
        "anthropic",
    } and exception_type.__name__.endswith("BadRequestError"):
        return True

    for module_name in ("openai", "anthropic"):
        try:
            module = importlib.import_module(module_name)
            error_type = getattr(module, "BadRequestError", None)
        except (ImportError, AttributeError):
            continue
        if isinstance(error_type, type) and isinstance(exc, error_type):
            return True

    if getattr(exc, "status_code", None) == 400:
        return True

    error_text = " ".join(
        str(value)
        for value in (
            exc,
            getattr(exc, "body", None),
            getattr(exc, "message", None),
        )
        if value is not None
    )
    return "invalid_request_error" in error_text.lower()


class _ProviderValidationAwareRunnableRetry(RunnableRetry):
    @property
    def _kwargs_retrying(self) -> dict[str, object]:
        kwargs = super()._kwargs_retrying
        kwargs["retry"] = retry_if_exception(
            lambda exception: not is_provider_validation_error(exception)
        )
        return kwargs


class ProviderValidationAwareModelFallbackMiddleware(ModelFallbackMiddleware):
    """Stop fallback when the provider rejects the request shape."""

    def wrap_model_call(self, request, handler):
        """Run fallback without swallowing provider validation errors."""
        validation_error: Exception | None = None

        def guarded_handler(current_request):
            nonlocal validation_error
            try:
                return handler(current_request)
            except Exception as exc:
                if not is_provider_validation_error(exc):
                    raise
                validation_error = exc
                raise GraphBubbleUp()

        try:
            return super().wrap_model_call(request, guarded_handler)
        except GraphBubbleUp:
            if validation_error is not None:
                raise validation_error
            raise

    async def awrap_model_call(self, request, handler):
        """Run async fallback without swallowing provider validation errors."""
        validation_error: Exception | None = None

        async def guarded_handler(current_request):
            nonlocal validation_error
            try:
                return await handler(current_request)
            except Exception as exc:
                if not is_provider_validation_error(exc):
                    raise
                validation_error = exc
                raise GraphBubbleUp()

        try:
            return await super().awrap_model_call(request, guarded_handler)
        except GraphBubbleUp:
            if validation_error is not None:
                raise validation_error
            raise


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
                if is_provider_validation_error(e):
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
    "ProviderValidationAwareModelFallbackMiddleware",
    "is_provider_validation_error",
]
