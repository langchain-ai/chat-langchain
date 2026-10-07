"""Mark superseded user turns only in answering-model requests."""

from collections.abc import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import HumanMessage

_SUPERSEDED_PREFIX = (
    "[Earlier message from a stopped turn—the user moved on; "
    "use only as context if the latest message refers to it]"
)


class StaleTurnMiddleware(AgentMiddleware):
    """Keep unanswered earlier user messages from hijacking the current answer."""

    def _annotate_stale_turns(self, request: ModelRequest) -> ModelRequest:
        messages = request.messages
        latest_index = next(
            (
                index
                for index in range(len(messages) - 1, -1, -1)
                if isinstance(messages[index], HumanMessage)
            ),
            None,
        )
        if latest_index is None:
            return request

        first_index = latest_index
        while first_index > 0 and isinstance(messages[first_index - 1], HumanMessage):
            first_index -= 1
        if first_index == latest_index:
            return request

        annotated_messages = list(messages)
        for index in range(first_index, latest_index):
            message = messages[index].model_copy(deep=True)
            if isinstance(message.content, str):
                message.content = f"{_SUPERSEDED_PREFIX}\n{message.content}"
            else:
                message.content = [
                    {"type": "text", "text": _SUPERSEDED_PREFIX},
                    *message.content,
                ]
            annotated_messages[index] = message
        return request.override(messages=annotated_messages)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Annotate superseded turns before a synchronous answering-model call."""
        return handler(self._annotate_stale_turns(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Annotate superseded turns before an asynchronous answering-model call."""
        return await handler(self._annotate_stale_turns(request))
