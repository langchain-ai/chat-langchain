"""Retry malformed, truncated, or degenerate model responses with backoff."""

import asyncio
import logging
import time
import zlib
from typing import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)

# Finish reasons that indicate a retryable failure (not an exception)
RETRYABLE_FINISH_REASONS = {
    "MALFORMED_FUNCTION_CALL",  # Gemini: invalid tool call syntax
    "MAX_TOKENS",
    "length",
}


def _is_degenerate_output(text: str) -> bool:
    if len(text) <= 4_000:
        return False
    raw = text.encode("utf-8")
    return len(zlib.compress(raw)) < len(raw) * 0.05


def get_retryable_response_reason(response: object) -> str | None:
    """Identify retryable finish reasons or highly repetitive response text."""
    messages = response.result if isinstance(response, ModelResponse) else [response]
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        finish_reason = message.response_metadata.get("finish_reason", "")
        if finish_reason in RETRYABLE_FINISH_REASONS:
            return finish_reason
        if _is_degenerate_output(message.text):
            return "degenerate output"
    return None


class MalformedResponseError(Exception):
    """Raised when model returns a malformed response after exhausting retries."""

    pass


class ModelRetryMiddleware(AgentMiddleware):
    """Bound model output and retry unusable responses before falling back."""

    def __init__(
        self,
        max_retries: int = 2,
        initial_delay: float = 0.5,
        backoff_factor: float = 2.0,
        max_output_tokens: int = 8192,
    ):
        """Configure the retry budget, backoff, and output limit."""
        super().__init__()
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.max_output_tokens = max_output_tokens

    def _bounded_request(self, request: ModelRequest) -> ModelRequest:
        setting = (
            "max_output_tokens"
            if isinstance(request.model, ChatGoogleGenerativeAI)
            else "max_tokens"
        )
        return request.override(
            model_settings={
                **request.model_settings,
                setting: self.max_output_tokens,
            }
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Bound and retry synchronous model calls."""
        request = self._bounded_request(request)
        for attempt in range(self.max_retries + 1):
            try:
                response = handler(request)
                reason = get_retryable_response_reason(response)
                if reason:
                    raise MalformedResponseError(f"Model returned {reason}")
                return response
            except Exception:
                if attempt == self.max_retries:
                    raise
                time.sleep(self.initial_delay * (self.backoff_factor**attempt))
        raise RuntimeError("Unexpected state in retry middleware")

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Bound and retry asynchronous model calls."""
        request = self._bounded_request(request)
        last_exception: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = await handler(request)
                reason = get_retryable_response_reason(response)
                if reason:
                    raise MalformedResponseError(f"Model returned {reason}")

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
