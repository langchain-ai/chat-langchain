# Middleware to normalize non-consecutive system messages for Anthropic compatibility
import logging
from typing import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


class MessageNormalizationMiddleware(AgentMiddleware):
    """Normalize non-consecutive system messages before passing to the model.

    Anthropic models raise ``ValueError('Received multiple non-consecutive
    system messages.')`` when the message list contains a ``SystemMessage``
    after any non-system message (e.g. a per-turn context message injected by
    the frontend).

    This middleware converts any such out-of-position ``SystemMessage`` to a
    ``HumanMessage`` with a ``[Context: ...]`` prefix so that Anthropic (and
    every other provider) receives a well-formed conversation history.

    The transformation is a no-op for single-turn conversations or any
    conversation that does not contain non-consecutive system messages.
    """

    @staticmethod
    def _normalize_messages(messages: list[AnyMessage]) -> list[AnyMessage]:
        """Return a new message list with non-consecutive system messages converted.

        A ``SystemMessage`` is considered *consecutive* only if it appears
        before any non-system message has been seen.  The very first block of
        ``SystemMessage`` items (index 0, 1, …) is kept as-is; every subsequent
        ``SystemMessage`` is converted to a ``HumanMessage``.

        Args:
            messages: The conversation history from the model request.

        Returns:
            A (possibly identical) list of messages.  The original list is
            returned unchanged when no conversion is needed to avoid
            unnecessary object creation on the hot path.
        """
        seen_non_system = False
        needs_normalization = False

        for msg in messages:
            if isinstance(msg, SystemMessage):
                if seen_non_system:
                    needs_normalization = True
                    break
            else:
                seen_non_system = True

        if not needs_normalization:
            return messages

        normalized: list[AnyMessage] = []
        seen_non_system = False
        for msg in messages:
            if isinstance(msg, SystemMessage) and seen_non_system:
                # Convert out-of-position system message to human context
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                normalized.append(HumanMessage(content=f"[Context: {content}]"))
                logger.debug(
                    "Converted non-consecutive SystemMessage to HumanMessage: %.80s…",
                    content,
                )
            else:
                if not isinstance(msg, SystemMessage):
                    seen_non_system = True
                normalized.append(msg)

        return normalized

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        normalized = self._normalize_messages(request.messages)
        if normalized is not request.messages:
            request = request.override(messages=normalized)
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        normalized = self._normalize_messages(request.messages)
        if normalized is not request.messages:
            request = request.override(messages=normalized)
        return await handler(request)


__all__ = ["MessageNormalizationMiddleware"]
