"""Stop runaway tool loops before they exhaust the graph recursion limit."""

from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

_TOOL_BUDGET_MESSAGE = (
    "The per-turn tool-call budget has been reached. Finalize your answer using "
    "the results already gathered instead of executing another tool."
)


class ToolCallBudgetMiddleware(AgentMiddleware):
    """Limit tool calls in one human turn using graph message state."""

    def __init__(self, max_tool_calls: int = 12) -> None:
        """Initialize the maximum number of tool calls per turn."""
        self.max_tool_calls = max_tool_calls

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Return a finalize instruction after the per-turn budget is spent."""
        if self._tool_call_number(request) > self.max_tool_calls:
            return ToolMessage(
                content=_TOOL_BUDGET_MESSAGE,
                name=request.tool_call.get("name", "unknown_tool"),
                tool_call_id=request.tool_call.get("id", ""),
            )
        return await handler(request)

    def _tool_call_number(self, request: ToolCallRequest) -> int:
        messages = self._messages(request.state)
        human_index = self._last_human_index(messages)
        call_number = 0
        current_call_id = request.tool_call.get("id")

        for message in messages[human_index + 1 :]:
            tool_calls = getattr(message, "tool_calls", None) or []
            for tool_call in tool_calls:
                call_number += 1
                if tool_call.get("id") == current_call_id:
                    return call_number

        return call_number + 1

    def _last_human_index(self, messages: list[BaseMessage]) -> int:
        for index in range(len(messages) - 1, -1, -1):
            if getattr(messages[index], "type", None) == "human":
                return index
        return -1

    def _messages(self, state: Any) -> list[BaseMessage]:
        if isinstance(state, Mapping):
            messages = state.get("messages", [])
        else:
            messages = getattr(state, "messages", state or [])
        return list(messages)
