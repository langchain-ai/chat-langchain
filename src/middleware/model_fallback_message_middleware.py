"""Terminal fallback that turns surviving model-provider errors into a reply."""

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

DEFAULT_FALLBACK_MESSAGE = (
    "The model provider had a temporary error; please retry."
)

RETRYABLE_STATUS_CODES = {500, 502, 503, 504}

RETRYABLE_ERROR_MARKERS = (
    "apierror",
    "bad gateway",
    "connection error",
    "connection reset",
    "gateway time-out",
    "gateway timeout",
    "help.openai.com",
    "internal server error",
    "overloaded",
    "request id req_",
    "server error",
    "service unavailable",
    "temporarily unavailable",
    "timed out",
    "timeout",
)


class ModelErrorFallbackMiddleware(AgentMiddleware):
    """Return a user-facing message when a provider error survives retries."""

    def __init__(self, message: str = DEFAULT_FALLBACK_MESSAGE):
        super().__init__()
        self.message = message

    def _status_code(self, error: Exception) -> int | None:
        status_code = getattr(error, "status_code", None)
        if isinstance(status_code, int):
            return status_code

        response = getattr(error, "response", None)
        response_status = getattr(response, "status_code", None)
        if isinstance(response_status, int):
            return response_status

        return None

    def _is_transient(self, error: Exception) -> bool:
        if self._status_code(error) in RETRYABLE_STATUS_CODES:
            return True

        text = (str(error) or error.__class__.__name__).lower()
        return any(marker in text for marker in RETRYABLE_ERROR_MARKERS)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        try:
            return await handler(request)
        except Exception as error:
            if not self._is_transient(error):
                raise
            logger.error(
                f"Model call failed with transient provider error, "
                f"returning fallback message: {error}"
            )
            return AIMessage(content=self.message)


__all__ = ["ModelErrorFallbackMiddleware", "DEFAULT_FALLBACK_MESSAGE"]
