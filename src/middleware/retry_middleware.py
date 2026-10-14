"""Retry middleware for model calls and provider responses."""

import asyncio
import logging
import re
import time
from typing import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables.retry import RunnableRetry
from tenacity import retry_if_exception

logger = logging.getLogger(__name__)

# Finish reasons that indicate a retryable failure (not an exception)
RETRYABLE_FINISH_REASONS = {
    "MALFORMED_FUNCTION_CALL",  # Gemini: invalid tool call syntax
    "MAX_TOKENS",
    "length",
    "max_tokens",
}
RECOVERY_MESSAGE = "I couldn't generate an answer, please rephrase."
REPEATED_UNIT = re.compile(r"(.{1,8}?)\1{49,}", re.DOTALL)


class MalformedResponseError(Exception):
    """Signal malformed, truncated, or repetitive model output."""

    pass


def validate_model_response(
    response: ModelResponse | AIMessage,
) -> ModelResponse | AIMessage:
    """Reject truncated or long, low-information model output."""
    messages = response.result if isinstance(response, ModelResponse) else [response]
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        metadata = message.response_metadata
        for key in ("finish_reason", "stop_reason"):
            if metadata.get(key) in RETRYABLE_FINISH_REASONS:
                raise MalformedResponseError(f"Model returned {metadata[key]}")
        content = message.content
        text = (
            content
            if isinstance(content, str)
            else "".join(
                block if isinstance(block, str) else block.get("text", "")
                for block in content
                if isinstance(block, str) or block.get("type") == "text"
            )
        )
        if len(text) >= 2000 and any(
            match.end() - match.start() >= len(text) * 0.8
            for match in REPEATED_UNIT.finditer(text)
        ):
            raise MalformedResponseError("Model returned repetitive output")
    return response


class ModelRecoveryMiddleware(AgentMiddleware):
    """Bound the runtime primary and recover after the fallback chain fails."""

    def __init__(self, primary_model: BaseChatModel):
        """Configure the bounded primary while preserving the deployment model literal."""
        super().__init__()
        self.primary_model = primary_model

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Return only accepted output or an explicit recovery message."""
        try:
            return handler(request.override(model=self.primary_model))
        except Exception:
            logger.exception("All answer models failed")
            return ModelResponse(result=[AIMessage(content=RECOVERY_MESSAGE)])

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Return only accepted output or an explicit recovery message."""
        try:
            return await handler(request.override(model=self.primary_model))
        except Exception:
            logger.exception("All answer models failed")
            return ModelResponse(result=[AIMessage(content=RECOVERY_MESSAGE)])


class _ProviderValidationAwareRunnableRetry(RunnableRetry):
    @property
    def _kwargs_retrying(self) -> dict[str, object]:
        kwargs = super()._kwargs_retrying
        kwargs["retry"] = retry_if_exception(
            lambda exception: not isinstance(exception, ValueError)
        )
        return kwargs


class ModelRetryMiddleware(AgentMiddleware):
    """Retry transient model failures and malformed responses."""

    def __init__(
        self,
        max_retries: int = 1,
        initial_delay: float = 0.5,
        backoff_factor: float = 2.0,
    ):
        """Configure retry attempts and backoff timing."""
        super().__init__()
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Retry transient failures and validate every attempt, including the last."""
        for attempt in range(self.max_retries + 1):
            try:
                response = handler(request)
                validate_model_response(response)
                return response
            except Exception as exception:
                if isinstance(exception, ValueError) or attempt == self.max_retries:
                    raise
                logger.warning("Model call failed; retrying: %s", exception)
                time.sleep(self.initial_delay * (self.backoff_factor**attempt))
        raise RuntimeError("Unexpected state in retry middleware")

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Retry transient failures from the wrapped model handler."""
        for attempt in range(self.max_retries + 1):
            try:
                response = await handler(request)
                validate_model_response(response)
                return response

            except Exception as e:
                if isinstance(e, ValueError) or attempt == self.max_retries:
                    raise
                delay = self.initial_delay * (self.backoff_factor**attempt)
                logger.warning(
                    f"Model call failed attempt {attempt + 1}/{self.max_retries + 1}: {e}, "
                    f"retrying in {delay:.2f}s"
                )
                await asyncio.sleep(delay)
        raise RuntimeError("Unexpected state in retry middleware")


__all__ = ["ModelRetryMiddleware", "ModelRecoveryMiddleware", "MalformedResponseError"]
