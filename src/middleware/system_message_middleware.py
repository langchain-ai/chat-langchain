# Middleware to merge non-consecutive system messages for Anthropic API compatibility
import logging
from typing import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import SystemMessage

logger = logging.getLogger(__name__)


class SystemMessageMiddleware(AgentMiddleware):
    """Merge non-consecutive system messages into a single system message.

    Anthropic's API rejects messages arrays that contain multiple system
    messages at non-consecutive positions (raises ValueError). This occurs
    in multi-turn conversations where each turn's page context is stored as
    a SystemMessage in conversation history, resulting in interleaved system
    messages between human/AI turns.

    This middleware consolidates all SystemMessage instances into a single
    SystemMessage at position 0, concatenating their content with newlines,
    before the request reaches the model.
    """

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Merge non-consecutive system messages before passing to the model."""
        messages = list(request.messages)

        system_messages = [m for m in messages if isinstance(m, SystemMessage)]

        if len(system_messages) <= 1:
            # Nothing to merge — pass through as-is
            return await handler(request)

        # Check if there are non-consecutive system messages
        # (i.e., system messages not all at the start)
        system_indices = [i for i, m in enumerate(messages) if isinstance(m, SystemMessage)]
        max_consecutive_start = 0
        for i, idx in enumerate(system_indices):
            if idx == i:
                max_consecutive_start = i + 1
            else:
                break

        if max_consecutive_start == len(system_indices):
            # All system messages are already consecutive at the start — pass through
            return await handler(request)

        # Merge all system messages into one at position 0
        merged_content = "\n\n".join(
            m.content if isinstance(m.content, str) else str(m.content)
            for m in system_messages
        )

        logger.info(
            "Merging %d non-consecutive system messages into one "
            "(positions: %s)",
            len(system_messages),
            system_indices,
        )

        merged_system = SystemMessage(content=merged_content)
        non_system_messages = [m for m in messages if not isinstance(m, SystemMessage)]
        cleaned_messages = [merged_system] + non_system_messages

        request = request.model_copy(update={"messages": cleaned_messages})
        return await handler(request)


__all__ = ["SystemMessageMiddleware"]
