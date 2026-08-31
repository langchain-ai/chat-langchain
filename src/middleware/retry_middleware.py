# Retry middleware for model calls with exponential backoff
import asyncio
import logging
from typing import Awaitable, Callable

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

    def _normalize_content(self, response: ModelResponse) -> ModelResponse:
        """Normalize list content from Gemini models to a plain string.

        ChatGoogleGenerativeAI with Gemini 3.x returns AIMessage.content as a
        list of content-part dicts (e.g. [{"text": "...", "extras": {...}}]).
        Only normalize when every block is a plain text block — never touch
        lists that contain "thinking" or "tool_use" blocks, which are
        legitimate structured outputs.
        """
        content = getattr(response, "content", None)
        if not isinstance(content, list) or not content:
            return response

        texts: list[str] = []
        for block in content:
            if isinstance(block, str):
                texts.append(block)
            elif isinstance(block, dict):
                block_type = block.get("type")
                if block_type in ("thinking", "tool_use"):
                    # Structured output — do not normalize.
                    return response
                text = block.get("text")
                if text is None:
                    # Block has no text key; leave as-is.
                    return response
                texts.append(text)
            else:
                return response

        return response.model_copy(update={"content": "".join(texts)})

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
                response = self._normalize_content(response)
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


__all__ = ["ModelRetryMiddleware", "MalformedResponseError"]
