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
from langchain_core.messages import AIMessage
from langchain_core.runnables.retry import RunnableRetry
from tenacity import retry_if_exception

logger = logging.getLogger(__name__)

# Finish reasons that indicate a retryable failure (not an exception)
RETRYABLE_FINISH_REASONS = {
    "MALFORMED_FUNCTION_CALL",  # Gemini: invalid tool call syntax
}


def is_non_retryable_model_error(exception: Exception) -> bool:
    """Return whether a model error should not be retried."""
    exception_name = type(exception).__name__
    return (
        isinstance(exception, ValueError)
        or exception_name.endswith(("InvalidRequestError", "BadRequestError"))
        or getattr(exception, "status_code", None) in (400, 422)
        or getattr(exception, "http_status", None) in (400, 422)
    )


class MalformedResponseError(Exception):
    """Raised when model returns a malformed response after exhausting retries."""

    pass


class _ProviderValidationAwareRunnableRetry(RunnableRetry):
    @property
    def _kwargs_retrying(self) -> dict[str, object]:
        kwargs = super()._kwargs_retrying
        kwargs["retry"] = retry_if_exception(
            lambda exception: not is_non_retryable_model_error(exception)
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
                if is_non_retryable_model_error(e):
                    logger.error("Model call failed with a non-retryable error: %s", e)
                    return ModelResponse(
                        result=[
                            AIMessage(
                                content="I couldn't generate an answer right now."
                            )
                        ]
                    )
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

        if last_exception:
            return ModelResponse(
                result=[AIMessage(content="I couldn't generate an answer right now.")]
            )

        if last_retryable_reason:
            logger.error(
                "Model returned %s after %s attempts",
                last_retryable_reason,
                self.max_retries + 1,
            )
            return ModelResponse(
                result=[AIMessage(content="I couldn't generate an answer right now.")]
            )

        raise RuntimeError("Unexpected state in retry middleware")


__all__ = [
    "ModelRetryMiddleware",
    "MalformedResponseError",
    "is_non_retryable_model_error",
]
