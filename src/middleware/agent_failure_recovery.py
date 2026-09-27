"""Recover failed agent turns without leaving unanswered user messages."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.middleware.types import (
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, HumanMessage

RECOVERY_MESSAGE = "I couldn’t complete this turn. Please try again."
MAX_CHECK_LINKS_CALLS = 8


class AgentFailureRecoveryMiddleware(AgentMiddleware[AgentState]):
    """Terminate failed or repeating turns with one assistant message."""

    def before_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        """Stop an uncapped check-links loop before graph recursion is exhausted."""
        if self._check_links_calls(state.get("messages", [])) < MAX_CHECK_LINKS_CALLS:
            return None
        return {"messages": [AIMessage(content=RECOVERY_MESSAGE)], "jump_to": "end"}

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Convert model failures into a committed terminal assistant message."""
        try:
            return await handler(request)
        except Exception:
            return ModelResponse(result=[AIMessage(content=RECOVERY_MESSAGE)])

    def _check_links_calls(self, messages: list[Any]) -> int:
        latest_human = next(
            (
                index
                for index in range(len(messages) - 1, -1, -1)
                if isinstance(messages[index], HumanMessage)
            ),
            -1,
        )
        return sum(
            1
            for message in messages[latest_human + 1 :]
            if isinstance(message, AIMessage)
            and message.tool_calls
            and all(call.get("name") == "check_links" for call in message.tool_calls)
        )


__all__ = [
    "AgentFailureRecoveryMiddleware",
    "MAX_CHECK_LINKS_CALLS",
    "RECOVERY_MESSAGE",
]
