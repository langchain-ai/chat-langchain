"""Convert deterministic provider failures into delivered model responses."""

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import (
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, HumanMessage

from src.middleware.retry_middleware import _is_non_retryable

logger = logging.getLogger(__name__)

_FAILURE_NOTICE = (
    "I couldn't complete that request because the model rejected its request "
    "format. Please try again."
)


class ModelFailureBoundaryMiddleware(AgentMiddleware):
    """Return a user-facing response for deterministic provider failures."""

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Convert non-retryable model failures into an AI response."""
        try:
            return await handler(request)
        except Exception as exc:
            if not _is_non_retryable(exc):
                raise
            logger.exception("Model request rejected after fallback chain", exc_info=exc)
            return ModelResponse(result=[AIMessage(content=self._failure_content(request))])

    def _failure_content(self, request: ModelRequest) -> str:
        """Return the latest non-empty AI draft or a failure notice."""
        for message in reversed(request.messages):
            if isinstance(message, HumanMessage):
                break
            if isinstance(message, AIMessage):
                content = self._content_text(message.content)
                if content:
                    return content
        return _FAILURE_NOTICE

    def _content_text(self, content: Any) -> str:
        """Extract displayable text from message content."""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and isinstance(item.get("text"), str)
            ).strip()
        return str(content).strip() if content else ""


__all__ = ["ModelFailureBoundaryMiddleware"]
