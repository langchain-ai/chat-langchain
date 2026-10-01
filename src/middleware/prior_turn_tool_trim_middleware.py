"""Trim tool exchanges from completed turns before model calls."""

from collections.abc import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import (
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage


class PriorTurnToolTrimMiddleware(AgentMiddleware):
    """Remove prior-turn tool exchanges while preserving current-turn context."""

    def _trim_messages(self, messages: list[AnyMessage]) -> list[AnyMessage]:
        latest_human_index = next(
            (
                index
                for index in range(len(messages) - 1, -1, -1)
                if isinstance(messages[index], HumanMessage)
            ),
            -1,
        )
        trimmed: list[AnyMessage] = []
        for index, message in enumerate(messages):
            if index > latest_human_index:
                trimmed.append(message)
                continue
            if isinstance(message, ToolMessage):
                continue
            if isinstance(message, AIMessage) and message.tool_calls:
                if not message.text.strip():
                    continue
                message = message.model_copy(update={"tool_calls": []})
            trimmed.append(message)
        return trimmed

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Trim completed turns before the synchronous model call."""
        return handler(request.override(messages=self._trim_messages(request.messages)))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Trim completed turns before the asynchronous model call."""
        return await handler(
            request.override(messages=self._trim_messages(request.messages))
        )


__all__ = ["PriorTurnToolTrimMiddleware"]
