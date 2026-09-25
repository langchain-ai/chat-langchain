"""Short-circuit duplicate tool calls within a graph run."""

import json
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

_RUN_STATE_KEY = "_duplicate_tool_call_middleware_state"
_CHECK_LINKS_BUDGET = 5
_DUPLICATE_DIRECTIVE = (
    "You already made this exact call and the result has not changed. "
    "Do not call it again. Proceed to your final answer."
)
_CHECK_LINKS_BUDGET_MESSAGE = (
    "Link validation budget exhausted. Write your final answer now using only "
    "URLs already confirmed valid."
)


class DuplicateToolCallMiddleware(AgentMiddleware[AgentState]):
    """Prevent duplicate tool calls and bound link validation per graph run."""

    def __init__(self):
        """Initialize per-run fallback state."""
        super().__init__()
        self._fallback_states: dict[str, dict[str, Any]] = {}

    def _run_state(self, request: ToolCallRequest) -> dict[str, Any]:
        context = request.runtime.context
        if isinstance(context, dict):
            return context.setdefault(
                _RUN_STATE_KEY,
                {"calls": {}, "check_links_count": 0},
            )

        run_id = request.runtime.config.get("run_id")
        state_key = str(run_id or id(context))
        return self._fallback_states.setdefault(
            state_key,
            {"calls": {}, "check_links_count": 0},
        )

    def _tool_message(self, request: ToolCallRequest, content: str) -> ToolMessage:
        return ToolMessage(
            content=content,
            name=request.tool_call.get("name", "unknown_tool"),
            tool_call_id=request.tool_call.get("id", ""),
        )

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler,
    ) -> ToolMessage | Command:
        """Short-circuit repeated calls and enforce link validation limits."""
        state = self._run_state(request)
        tool_call = request.tool_call
        tool_name = tool_call.get("name", "unknown_tool")
        signature = (
            tool_name,
            json.dumps(tool_call.get("args", {}), sort_keys=True),
        )
        cached_content = state["calls"].get(signature)
        if cached_content is not None:
            return self._tool_message(
                request,
                f"{cached_content}\n\n{_DUPLICATE_DIRECTIVE}",
            )

        if tool_name == "check_links":
            if state["check_links_count"] >= _CHECK_LINKS_BUDGET:
                return self._tool_message(request, _CHECK_LINKS_BUDGET_MESSAGE)
            state["check_links_count"] += 1

        result = await handler(request)
        if isinstance(result, ToolMessage):
            state["calls"][signature] = result.content
        return result
