"""Disambiguate unanswered user messages without changing thread state."""

from collections.abc import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
)
from langchain_core.messages import HumanMessage

_SUPERSEDED = (
    "[Cancelled or superseded user request: context only, not a question to answer.]"
)
_CURRENT = (
    "[Current user request: answer only this message. Earlier consecutive user "
    "messages are cancelled or superseded context.]"
)


class LatestUserMessageMiddleware(AgentMiddleware):
    """Mark the latest unanswered user request on every model call."""

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelCallResult],
    ) -> ModelCallResult:
        """Pass a request-only copy to the synchronous model handler."""
        return handler(self._mark_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelCallResult]],
    ) -> ModelCallResult:
        """Pass a request-only copy to the asynchronous model handler."""
        return await handler(self._mark_request(request))

    def _mark_request(self, request: ModelRequest) -> ModelRequest:
        latest = next(
            (
                index
                for index in range(len(request.messages) - 1, -1, -1)
                if isinstance(request.messages[index], HumanMessage)
            ),
            None,
        )
        if latest is None:
            return request

        first = latest
        while first > 0 and isinstance(request.messages[first - 1], HumanMessage):
            first -= 1
        if first == latest:
            return request

        messages = list(request.messages)
        for index in range(first, latest + 1):
            message = messages[index]
            marker = _CURRENT if index == latest else _SUPERSEDED
            content = (
                f"{marker}\n\n{message.content}"
                if isinstance(message.content, str)
                else [{"type": "text", "text": marker}, *message.content]
            )
            messages[index] = message.model_copy(update={"content": content})
        return request.override(messages=messages)


__all__ = ["LatestUserMessageMiddleware"]
