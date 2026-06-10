"""Convert uncaught model-loop exceptions into a user-visible apology AIMessage."""
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
    "Sorry, I ran into a problem looking that up. "
    "Please try rephrasing your question or retrying in a moment."
)


class GracefulErrorMiddleware(AgentMiddleware):
    """Outermost model-call wrapper that turns exceptions into a fallback AIMessage."""

    def __init__(self, fallback_message: str = DEFAULT_FALLBACK_MESSAGE):
        """Initialize with the AIMessage content used on uncaught failures."""
        super().__init__()
        self.fallback_message = fallback_message

    def _fallback_response(self) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content=self.fallback_message)])

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Run the model call and return a fallback AIMessage if it raises."""
        try:
            return await handler(request)
        except Exception as error:
            logger.exception(
                "Model loop failed; returning fallback AIMessage: %s", error
            )
            return self._fallback_response()

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Run the model call and return a fallback AIMessage if it raises."""
        try:
            return handler(request)
        except Exception as error:
            logger.exception(
                "Model loop failed; returning fallback AIMessage: %s", error
            )
            return self._fallback_response()


__all__ = ["GracefulErrorMiddleware", "DEFAULT_FALLBACK_MESSAGE"]
