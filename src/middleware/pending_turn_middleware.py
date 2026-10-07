"""Direct pending human turns to the latest user message."""

from collections.abc import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import HumanMessage, SystemMessage

_PENDING_TURN_INSTRUCTIONS = (
    "The user sent earlier messages that were cancelled before an answer. "
    "Answer ONLY the final user message; use earlier ones as context only."
)


class PendingTurnMiddleware(AgentMiddleware):
    """Clarify the current question without changing checkpointed history."""

    def _prepare_request(self, request: ModelRequest) -> ModelRequest:
        if len(request.messages) < 2 or not all(
            isinstance(message, HumanMessage) for message in request.messages[-2:]
        ):
            return request

        system_message = request.system_message or SystemMessage(content="")
        content = system_message.content
        if isinstance(content, str):
            content = f"{content}\n\n{_PENDING_TURN_INSTRUCTIONS}".strip()
        else:
            content = [
                *content,
                {"type": "text", "text": _PENDING_TURN_INSTRUCTIONS},
            ]
        return request.override(
            system_message=system_message.model_copy(update={"content": content})
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Add pending-turn guidance to a synchronous model request."""
        return handler(self._prepare_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Add pending-turn guidance to an asynchronous model request."""
        return await handler(self._prepare_request(request))


__all__ = ["PendingTurnMiddleware"]
