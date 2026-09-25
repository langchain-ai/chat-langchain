# Retry middleware for model calls with exponential backoff
import asyncio
import logging
from collections.abc import Awaitable, Callable

import langsmith as ls
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
        last_response: ModelResponse | None = None

        for attempt in range(self.max_retries + 1):
            try:
                with ls.trace(
                    name="ModelRetryMiddleware.attempt",
                    run_type="chain",
                    inputs={"attempt": attempt + 1},
                    metadata={
                        "attempt": attempt + 1,
                        "max_attempts": self.max_retries + 1,
                    },
                ):
                    response = await handler(request)
                last_response = response
                finish_reason = self._get_finish_reason(response)

                if finish_reason in RETRYABLE_FINISH_REASONS:
                    last_retryable_reason = finish_reason
                    if attempt < self.max_retries:
                        delay = self.initial_delay * (self.backoff_factor**attempt)
                        logger.warning(
                            f"Retryable response ({finish_reason}) "
                            f"attempt {attempt + 1}/{self.max_retries + 1}, "
                            f"retrying in {delay:.2f}s"
                        )
                        await asyncio.sleep(delay)
                        continue

                return response

            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay = self.initial_delay * (self.backoff_factor**attempt)
                    logger.warning(
                        "Model call failed attempt %s/%s: %s, retrying in %.2fs",
                        attempt + 1,
                        self.max_retries + 1,
                        e,
                        delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "Model call failed after %s attempts: %s",
                        self.max_retries + 1,
                        e,
                    )

        # Exhausted retries - raise for fallback middleware
        if last_exception:
            raise last_exception

        if last_retryable_reason and last_response is not None:
            return last_response

        raise RuntimeError("Unexpected state in retry middleware")


__all__ = ["ModelRetryMiddleware", "MalformedResponseError"]
