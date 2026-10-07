"""Disambiguate the current question after stopped or cancelled turns."""

from collections.abc import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

_CONTEXT_LABEL = (
    "Earlier messages sent without an answer "
    "(context only; superseded unless the current question refers to them):"
)
_CURRENT_LABEL = "Current question to answer:"


class OrphanHumanMessageMiddleware(AgentMiddleware):
    """Mark unanswered earlier questions as context without changing stored history."""

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Pass the model a request targeting the latest human message."""
        return handler(self._prepare_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Pass the model a request targeting the latest human message."""
        return await handler(self._prepare_request(request))

    def _prepare_request(self, request: ModelRequest) -> ModelRequest:
        latest_index = next(
            (
                index
                for index in range(len(request.messages) - 1, -1, -1)
                if isinstance(request.messages[index], HumanMessage)
            ),
            None,
        )
        if latest_index is None:
            return request

        orphan_indices = []
        for index in range(latest_index - 1, -1, -1):
            message = request.messages[index]
            if isinstance(message, AIMessage) and message.text.strip():
                break
            if isinstance(message, HumanMessage):
                orphan_indices.append(index)
        if not orphan_indices:
            return request

        orphan_indices.reverse()
        context = HumanMessage(
            content=_CONTEXT_LABEL
            + "\n\n"
            + "\n\n".join(
                f"{number}. {request.messages[index].text}"
                for number, index in enumerate(orphan_indices, start=1)
            )
        )
        latest = request.messages[latest_index]
        content = (
            f"{_CURRENT_LABEL}\n\n{latest.content}"
            if isinstance(latest.content, str)
            else [{"type": "text", "text": _CURRENT_LABEL}, *latest.content]
        )
        messages: list[AnyMessage] = []
        for index, message in enumerate(request.messages):
            if index == orphan_indices[0]:
                messages.append(context)
            elif index in orphan_indices:
                continue
            elif index == latest_index:
                messages.append(latest.model_copy(update={"content": content}))
            else:
                messages.append(message)
        return request.override(messages=messages)


__all__ = ["OrphanHumanMessageMiddleware"]
