"""Retry middleware for model calls and provider responses."""

import asyncio
import logging
from typing import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.runnables import RunnableWithFallbacks
from langchain_core.runnables.retry import RunnableRetry
from tenacity import retry_if_exception

logger = logging.getLogger(__name__)

# Finish reasons that indicate a retryable failure (not an exception)
RETRYABLE_FINISH_REASONS = {
    "MALFORMED_FUNCTION_CALL",  # Gemini: invalid tool call syntax
}


def is_non_retryable_request_error(exception: BaseException) -> bool:
    """Return whether an exception represents an invalid provider request."""
    return (
        isinstance(exception, ValueError)
        or getattr(exception, "status_code", None) == 400
    )


class MalformedResponseError(Exception):
    """Raised when model returns a malformed response after exhausting retries."""

    pass


class _ProviderValidationAwareRunnableRetry(RunnableRetry):
    @property
    def _kwargs_retrying(self) -> dict[str, object]:
        kwargs = super()._kwargs_retrying
        kwargs["retry"] = retry_if_exception(
            lambda exception: not is_non_retryable_request_error(exception)
        )
        return kwargs


class _ProviderValidationAwareRunnableWithFallbacks(RunnableWithFallbacks):
    def invoke(self, input, config=None, **kwargs):
        if self.exception_key is not None and not isinstance(input, dict):
            raise ValueError(
                "If 'exception_key' is specified then input must be a dictionary."
                f"However found a type of {type(input)} for input"
            )
        first_exception = None
        last_exception = None
        for runnable in self.runnables:
            try:
                if self.exception_key and last_exception is not None:
                    input[self.exception_key] = last_exception
                return runnable.invoke(input, config, **kwargs)
            except self.exceptions_to_handle as exception:
                if is_non_retryable_request_error(exception):
                    raise
                if first_exception is None:
                    first_exception = exception
                last_exception = exception
            except BaseException:
                raise
        if first_exception is None:
            raise RuntimeError("No error stored at end of fallbacks.")
        raise first_exception

    async def ainvoke(self, input, config=None, **kwargs):
        if self.exception_key is not None and not isinstance(input, dict):
            raise ValueError(
                "If 'exception_key' is specified then input must be a dictionary."
                f"However found a type of {type(input)} for input"
            )
        first_exception = None
        last_exception = None
        for runnable in self.runnables:
            try:
                if self.exception_key and last_exception is not None:
                    input[self.exception_key] = last_exception
                return await runnable.ainvoke(input, config, **kwargs)
            except self.exceptions_to_handle as exception:
                if is_non_retryable_request_error(exception):
                    raise
                if first_exception is None:
                    first_exception = exception
                last_exception = exception
            except BaseException:
                raise
        if first_exception is None:
            raise RuntimeError("No error stored at end of fallbacks.")
        raise first_exception


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
                if is_non_retryable_request_error(e):
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
    "is_non_retryable_request_error",
    "_ProviderValidationAwareRunnableRetry",
    "_ProviderValidationAwareRunnableWithFallbacks",
]
