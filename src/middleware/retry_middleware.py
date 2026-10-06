"""Retry middleware for model calls with exponential backoff."""

import asyncio
import logging
import os
import queue
import threading
import time
from typing import Awaitable, Callable, cast

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)

logger = logging.getLogger(__name__)

# Finish reasons that indicate a retryable failure (not an exception)
RETRYABLE_FINISH_REASONS = {
    "MALFORMED_FUNCTION_CALL",  # Gemini: invalid tool call syntax
}


class MalformedResponseError(Exception):
    """Raised when model returns a malformed response after exhausting retries."""

    pass


class ModelRetryMiddleware(AgentMiddleware):
    """Retry model calls and bound each attempt by a wall-clock timeout."""

    def __init__(
        self,
        max_retries: int = 2,
        initial_delay: float = 0.5,
        backoff_factor: float = 2.0,
        timeout_seconds: float | None = None,
    ):
        """Initialize retry and timeout settings."""
        super().__init__()
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.timeout_seconds = timeout_seconds or float(
            os.getenv("MODEL_TIMEOUT_SECONDS", "30")
        )

    def _get_finish_reason(self, response: ModelResponse) -> str:
        """Extract finish_reason from response metadata."""
        metadata = getattr(response, "response_metadata", None) or {}
        return metadata.get("finish_reason", "")

    def _invoke_sync_with_timeout(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        result_queue: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)

        def invoke() -> None:
            try:
                result_queue.put((True, handler(request)))
            except Exception as error:
                result_queue.put((False, error))

        threading.Thread(target=invoke, daemon=True).start()
        try:
            succeeded, result = result_queue.get(timeout=self.timeout_seconds)
        except queue.Empty as error:
            raise TimeoutError(
                f"Model call exceeded {self.timeout_seconds:.2f}s timeout"
            ) from error
        if succeeded:
            return cast(ModelResponse, result)
        raise cast(Exception, result)

    def _retry_delay(self, attempt: int) -> float:
        return self.initial_delay * (self.backoff_factor**attempt)

    def _log_retry(self, attempt: int, error: Exception) -> None:
        delay = self._retry_delay(attempt)
        logger.warning(
            f"Model call failed attempt {attempt + 1}/{self.max_retries + 1}: {error}, "
            f"retrying in {delay:.2f}s"
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Retry synchronous model calls within the configured timeout."""
        last_exception: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._invoke_sync_with_timeout(request, handler)
                finish_reason = self._get_finish_reason(response)
                if finish_reason in RETRYABLE_FINISH_REASONS:
                    if attempt < self.max_retries:
                        delay = self._retry_delay(attempt)
                        logger.warning(
                            f"Retryable response ({finish_reason}) "
                            f"attempt {attempt + 1}/{self.max_retries + 1}, "
                            f"retrying in {delay:.2f}s"
                        )
                        time.sleep(delay)
                        continue
                    raise MalformedResponseError(
                        f"Model returned {finish_reason} after {self.max_retries + 1} attempts"
                    )
                return response
            except Exception as error:
                last_exception = error
                if attempt < self.max_retries:
                    self._log_retry(attempt, error)
                    time.sleep(self._retry_delay(attempt))
                else:
                    logger.error(
                        f"Model call failed after {self.max_retries + 1} attempts: {error}"
                    )
        raise last_exception or RuntimeError("Unexpected state in retry middleware")

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Retry asynchronous model calls within the configured timeout."""
        last_exception: Exception | None = None
        last_retryable_reason: str | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = await asyncio.wait_for(
                    handler(request), timeout=self.timeout_seconds
                )
                finish_reason = self._get_finish_reason(response)

                if finish_reason in RETRYABLE_FINISH_REASONS:
                    if attempt < self.max_retries:
                        delay = self._retry_delay(attempt)
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
                last_exception = e
                if attempt < self.max_retries:
                    self._log_retry(attempt, e)
                    delay = self._retry_delay(attempt)
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


__all__ = ["ModelRetryMiddleware", "MalformedResponseError"]
