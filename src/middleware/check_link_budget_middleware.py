"""Limit repeated link validation calls within an agent run."""

from collections.abc import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command


class CheckLinkBudgetMiddleware(AgentMiddleware[AgentState]):
    """Limit check_links calls and stop repeated validation loops."""

    def __init__(self, max_calls: int = 5):
        """Initialize the per-run call ceiling."""
        super().__init__()
        self.max_calls = max_calls

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Block check_links calls after the configured ceiling."""
        if request.tool_call.get("name") != "check_links":
            return await handler(request)

        call_count = request.state.get("_check_links_call_count", 0) + 1
        request.state["_check_links_call_count"] = call_count
        if call_count > self.max_calls:
            return ToolMessage(
                content=(
                    "The check_links budget is exhausted. Produce the final answer "
                    "using only URLs already reported valid, and do not call "
                    "check_links again."
                ),
                name="check_links",
                tool_call_id=request.tool_call.get("id", ""),
            )

        return await handler(request)


__all__ = ["CheckLinkBudgetMiddleware"]
