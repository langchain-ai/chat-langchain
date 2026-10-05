# Retry middleware for model calls with exponential backoff
import asyncio
import logging
import threading
from collections.abc import Awaitable, Callable

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


class ModelCallTimeoutMiddleware(AgentMiddleware):
    """Raise when a model call exceeds the configured timeout."""

    def __init__(self, timeout: float):
        """Initialize the model call timeout."""
        super().__init__()
        self.timeout = timeout

    @staticmethod
    def _model_name(request: ModelRequest) -> str:
        model = request.model
        return str(
            getattr(model, "model", None)
            or getattr(model, "model_name", None)
            or type(model).__name__
        )

    def _timeout_error(self, request: ModelRequest) -> TimeoutError:
        model_name = self._model_name(request)
        logger.warning(
            "Model call timed out after %.2fs: %s",
            self.timeout,
            model_name,
        )
        return TimeoutError(
            f"Model call timed out after {self.timeout:.2f}s: {model_name}"
        )

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Bound an asynchronous model call."""
        try:
            return await asyncio.wait_for(handler(request), timeout=self.timeout)
        except TimeoutError as exc:
            raise self._timeout_error(request) from exc

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Bound a synchronous model call."""
        result: list[ModelResponse] = []
        error: list[Exception] = []

        def run_handler() -> None:
            try:
                result.append(handler(request))
            except Exception as exc:
                error.append(exc)

        thread = threading.Thread(target=run_handler, daemon=True)
        thread.start()
        thread.join(timeout=self.timeout)
        if thread.is_alive():
            raise self._timeout_error(request)
        if error:
            raise error[0]
        return result[0]


class ModelRetryMiddleware(AgentMiddleware):
    def __init__(
        self,
        max_retries: int = 2,
        initial_delay: float = 0.5,
        backoff_factor: float = 2.0,
    ):
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
    "MalformedResponseError",
    "ModelCallTimeoutMiddleware",
    "ModelRetryMiddleware",
]
