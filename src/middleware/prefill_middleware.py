"""Middleware that prevents assistant-message model prefilling."""

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest
from langchain_core.messages import AIMessage, AnyMessage


class RemoveTrailingAIMessageMiddleware(AgentMiddleware):
    """Remove an assistant draft that would be sent as model prefill."""

    @staticmethod
    def _messages(request: ModelRequest) -> list[AnyMessage]:
        messages = request.messages
        if (
            messages
            and isinstance(messages[-1], AIMessage)
            and not messages[-1].tool_calls
        ):
            return messages[:-1]
        return messages

    def wrap_model_call(self, request, handler):
        """Remove a trailing assistant draft before model invocation."""
        return handler(request.override(messages=self._messages(request)))

    async def awrap_model_call(self, request, handler):
        """Remove a trailing assistant draft before model invocation."""
        return await handler(request.override(messages=self._messages(request)))


__all__ = ["RemoveTrailingAIMessageMiddleware"]
