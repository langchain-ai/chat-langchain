# Middleware to trim conversation history to fit within model context window.
"""Middleware to trim conversation history to fit within model context window."""

import logging
from typing import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import SystemMessage, trim_messages

logger = logging.getLogger(__name__)

# Target token budget — keep well below the 200k Claude limit to leave
# headroom for the system prompt, tool definitions, and the model's response.
MAX_TOKENS = 150_000

# Character-to-token ratio: Claude tokenises roughly 4 chars per token.
_CHARS_PER_TOKEN = 4


def _char_token_counter(messages: list) -> int:
    """Estimate token count using a character-based heuristic.

    Args:
        messages: List of LangChain message objects to count.

    Returns:
        Estimated number of tokens across all messages.
    """
    total = 0
    for msg in messages:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        total += len(content) // _CHARS_PER_TOKEN
    return total


class MessageTrimmerMiddleware(AgentMiddleware):
    """Trim conversation history before each model call to prevent token overflow.

    Uses a character-based token estimate (1 token ≈ 4 chars) and
    ``langchain_core.messages.trim_messages`` with ``strategy="last"`` so that
    the most recent messages are always preserved.  The system message is kept
    regardless of size.
    """

    def __init__(self, max_tokens: int = MAX_TOKENS):
        """Initialise the middleware with a configurable token budget.

        Args:
            max_tokens: Maximum number of tokens (character-estimated) to allow
                in the messages list passed to the model.
        """
        super().__init__()
        self.max_tokens = max_tokens
        logger.info(
            f"MessageTrimmerMiddleware initialised with max_tokens={max_tokens}"
        )

    def _trim_messages(self, messages: list) -> list:
        """Trim *messages* so that their estimated token count is within budget.

        Args:
            messages: Full conversation history.

        Returns:
            Trimmed message list, always preserving the system message and the
            most recent messages that fit within the budget.
        """
        if not messages:
            return messages

        current_tokens = _char_token_counter(messages)
        if current_tokens <= self.max_tokens:
            return messages

        logger.warning(
            f"Conversation history is ~{current_tokens} tokens "
            f"(>{self.max_tokens} budget). Trimming to fit context window."
        )

        # Detect whether a system message is present so trim_messages can
        # keep it even when applying the "last" strategy.
        has_system = any(isinstance(m, SystemMessage) for m in messages)

        trimmed = trim_messages(
            messages,
            strategy="last",
            max_tokens=self.max_tokens,
            token_counter=_char_token_counter,
            include_system=has_system,
            allow_partial=False,
        )

        trimmed_tokens = _char_token_counter(trimmed)
        removed = len(messages) - len(trimmed)
        logger.info(
            f"Trimmed {removed} message(s); "
            f"history now ~{trimmed_tokens} tokens "
            f"({len(trimmed)} messages)."
        )
        return trimmed

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Intercept the model call and trim messages before forwarding.

        Args:
            request: The incoming model request containing the message history.
            handler: The next handler in the middleware chain.

        Returns:
            The model response from the downstream handler.
        """
        request.messages = self._trim_messages(request.messages)
        return await handler(request)


__all__ = ["MessageTrimmerMiddleware", "MAX_TOKENS", "_char_token_counter"]
