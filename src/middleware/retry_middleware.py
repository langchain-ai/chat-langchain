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
from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)

# Finish reasons that indicate a retryable failure (not an exception)
RETRYABLE_FINISH_REASONS = {
    "MALFORMED_FUNCTION_CALL",  # Gemini: invalid tool call syntax
}


def _normalize_content(content) -> str:
    """Normalize message content to a plain string.

    Gemini models return content as a list of dicts (with thought signature
    metadata) instead of a plain string. Extract and join all text parts.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            item["text"]
            for item in content
            if isinstance(item, dict) and "text" in item
        ]
        return "".join(parts)
    return str(content)


def _normalize_response(response: ModelResponse) -> ModelResponse:
    """Normalize AI message content in a ModelResponse.

    If any AIMessage has list content (e.g. Gemini thought signatures),
    replace it with a plain string by extracting and joining text parts.
    """
    normalized_messages = []
    changed = False
    for msg in response.result:
        if isinstance(msg, AIMessage) and isinstance(msg.content, list):
            normalized = _normalize_content(msg.content)
            logger.debug(
                "Normalized Gemini list content to plain string "
                f"(had {len(msg.content)} part(s))"
            )
            msg = AIMessage(
                content=normalized,
                additional_kwargs=msg.additional_kwargs,
                response_metadata=msg.response_metadata,
                tool_calls=msg.tool_calls,
                id=msg.id,
            )
            changed = True
        normalized_messages.append(msg)
    if changed:
        return ModelResponse(
            result=normalized_messages,
            structured_response=response.structured_response,
        )
    return response


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

                return _normalize_response(response)

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
