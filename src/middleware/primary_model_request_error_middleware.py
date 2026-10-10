"""Report primary model request errors before fallback handles them."""

import logging
from collections.abc import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)

logger = logging.getLogger(__name__)


class PrimaryModelRequestErrorMiddleware(AgentMiddleware):
    """Log primary request validation failures without changing fallback behavior."""

    def __init__(self, primary_model: str):
        """Configure the primary model whose request errors should be reported."""
        super().__init__()
        self.primary_model = primary_model
        self.primary_model_name = primary_model.split(":", 1)[-1]

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Report primary validation errors and re-raise for the fallback wrapper."""
        try:
            return await handler(request)
        except (ValueError, TypeError) as exception:
            model_name = getattr(request.model, "model", None) or getattr(
                request.model, "model_name", None
            )
            if model_name == self.primary_model_name:
                logger.error(
                    "primary_model_request_error primary_model=%s: %s",
                    self.primary_model,
                    exception,
                )
            raise
