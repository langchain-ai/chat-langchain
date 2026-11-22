"""Clear large answered-turn tool payloads from model requests."""

from collections.abc import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage

CLEARED_TOOL_RESULT = (
    "[Earlier tool result cleared to save context. "
    "Re-run the tool if you need this content.]"
)


def _clear_stale_tool_results(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Copy large tool results preceding the latest user message."""
    last_human = next(
        (
            i
            for i in range(len(messages) - 1, -1, -1)
            if isinstance(messages[i], HumanMessage)
        ),
        0,
    )
    return [
        message.model_copy(update={"content": CLEARED_TOOL_RESULT})
        if i < last_human
        and isinstance(message, ToolMessage)
        and len(str(message.content)) > 500
        else message
        for i, message in enumerate(messages)
    ]


class StaleToolResultMiddleware(AgentMiddleware):
    """Keep large earlier tool results out of model input without changing state."""

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Clear stale results for a synchronous model call."""
        return handler(
            request.override(messages=_clear_stale_tool_results(request.messages))
        )

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Clear stale results for an asynchronous model call."""
        return await handler(
            request.override(messages=_clear_stale_tool_results(request.messages))
        )
