"""Mark unanswered user messages as superseded in model requests only."""

from collections.abc import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import HumanMessage

_SUPERSEDED_PREFIX = "[cancelled by user before a reply - superseded]\n"


class OrphanHumanMiddleware(AgentMiddleware):
    """Keep consecutive unanswered messages from competing with the latest question."""

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Mark superseded inputs before a synchronous model call."""
        return handler(self._prepare_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Mark superseded inputs before an asynchronous model call."""
        return await handler(self._prepare_request(request))

    def _prepare_request(self, request: ModelRequest) -> ModelRequest:
        messages = list(request.messages)
        changed = False
        for index, message in enumerate(request.messages[:-1]):
            if not isinstance(message, HumanMessage) or not isinstance(
                request.messages[index + 1], HumanMessage
            ):
                continue
            content = (
                _SUPERSEDED_PREFIX + message.content
                if isinstance(message.content, str)
                else [{"type": "text", "text": _SUPERSEDED_PREFIX}, *message.content]
            )
            messages[index] = message.model_copy(update={"content": content})
            changed = True
        return request.override(messages=messages) if changed else request


__all__ = ["OrphanHumanMiddleware"]
