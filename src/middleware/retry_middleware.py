"""Retry model calls with exponential backoff."""

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

logger = logging.getLogger(__name__)

# Finish reasons that indicate a retryable failure (not an exception)
RETRYABLE_FINISH_REASONS = {
    "MALFORMED_FUNCTION_CALL",  # Gemini: invalid tool call syntax
}


class MalformedResponseError(Exception):
    """Raised when model returns a malformed response after exhausting retries."""

    pass


class ModelRetryMiddleware(AgentMiddleware):
    """Retry exceptions and unusable model responses before fallback."""

    def __init__(
        self,
        max_retries: int = 2,
        initial_delay: float = 0.5,
        backoff_factor: float = 2.0,
    ):
        """Configure the retry budget and exponential backoff."""
        super().__init__()
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor

    def _last_ai_message(self, response: ModelResponse | AIMessage) -> AIMessage | None:
        """Return the last AI message from a wrapped or direct response."""
        if isinstance(response, AIMessage):
            return response
        return next(
            (
                message
                for message in reversed(response.result)
                if isinstance(message, AIMessage)
            ),
            None,
        )

    def _get_finish_reason(self, response: ModelResponse | AIMessage) -> str:
        """Extract finish_reason from the last AI message's metadata."""
        message = self._last_ai_message(response)
        metadata = message.response_metadata if message is not None else {}
        return metadata.get("finish_reason", "")

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse | AIMessage]],
    ) -> ModelCallResult:
        """Retry failed model calls and raise after exhaustion."""
        last_exception: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = await handler(request)
                finish_reason = self._get_finish_reason(response)

                message = self._last_ai_message(response)
                if finish_reason in RETRYABLE_FINISH_REASONS:
                    raise MalformedResponseError(f"Model returned {finish_reason}")
                if (
                    message is not None
                    and not message.tool_calls
                    and not message.text.strip()
                ):
                    raise MalformedResponseError("Model returned an empty response")

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

        raise RuntimeError("Unexpected state in retry middleware")


__all__ = ["ModelRetryMiddleware", "MalformedResponseError"]
