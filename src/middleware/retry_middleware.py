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

# Finish reasons that indicate the model hit its output-token ceiling and the
# assistant content is truncated (e.g. an unclosed code fence).
LENGTH_FINISH_REASONS = {"length", "max_tokens", "MAX_TOKENS"}

# Appended to a final assistant turn that was cut off so the user knows the
# answer is incomplete rather than silently receiving a half-formed code block.
TRUNCATION_MARKER = "\n\n*[Response was truncated. Ask me to continue.]*"

# Fence used to close a code block left open by a truncated response.
_CODE_FENCE = "```"


def _close_unterminated_code_fence(text: str) -> str:
    """Close a dangling triple-backtick fence so the block renders intact."""
    if text.count(_CODE_FENCE) % 2 == 1:
        suffix = "" if text.endswith("\n") else "\n"
        return f"{text}{suffix}{_CODE_FENCE}"
    return text


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

    def _final_ai_message(self, response: ModelResponse) -> AIMessage | None:
        """Return the final AIMessage produced by the model, if any."""
        if isinstance(response, AIMessage):
            return response
        result = getattr(response, "result", None) or []
        for message in reversed(result):
            if isinstance(message, AIMessage):
                return message
        return None

    def _get_finish_reason(self, response: ModelResponse) -> str:
        """Extract finish_reason from response metadata."""
        metadata = getattr(response, "response_metadata", None) or {}
        if not metadata:
            message = self._final_ai_message(response)
            metadata = getattr(message, "response_metadata", None) or {} if message else {}
        return metadata.get("finish_reason", "")

    def _mark_truncated_final_turn(self, response: ModelResponse) -> None:
        """Append a visible marker when a final assistant turn was cut off."""
        message = self._final_ai_message(response)
        if message is None or message.tool_calls:
            return
        if isinstance(message.content, str):
            if message.content.endswith(TRUNCATION_MARKER):
                return
            closed = _close_unterminated_code_fence(message.content)
            message.content = closed + TRUNCATION_MARKER
        elif isinstance(message.content, list):
            message.content.append({"type": "text", "text": TRUNCATION_MARKER})
        logger.warning("Final assistant turn truncated (finish_reason=length); appended marker")

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

                # A "length" finish reason means the output-token ceiling was
                # hit; retrying yields the same cutoff, so surface it to the user
                # instead of silently returning a half-formed answer.
                if finish_reason in LENGTH_FINISH_REASONS:
                    self._mark_truncated_final_turn(response)

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
    "ModelRetryMiddleware",
    "MalformedResponseError",
    "LENGTH_FINISH_REASONS",
    "TRUNCATION_MARKER",
]
