"""Keep an RL or eval trial from acting on live services.

The docs MCP's submit_feedback tool files feedback with the docs team. A trial reaches the live docs
service, so its policy's feedback would land there; with this middleware in the stack the call is
acknowledged without being sent. agent.py adds it when the agent runs as the RL policy
(TITO_POLICY_BASE_URL is set).
"""

from __future__ import annotations

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

STUBBED_TOOLS = ("submit_feedback",)
STUBBED_RESULT = "Feedback received. Thank you."


def is_stubbed(tool_name: str) -> bool:
    """Whether a tool is stubbed, by name or by an MCP server's prefixed name."""
    return any(
        tool_name == name or tool_name.endswith(f"_{name}") for name in STUBBED_TOOLS
    )


class TrialSideEffectsMiddleware(AgentMiddleware[AgentState]):
    """Acknowledges each call to a tool in STUBBED_TOOLS without running it."""

    def _stubbed(self, request: ToolCallRequest) -> ToolMessage | None:
        call = request.tool_call
        if not is_stubbed(call.get("name", "")):
            return None
        return ToolMessage(
            content=STUBBED_RESULT,
            tool_call_id=call.get("id", ""),
            name=call.get("name"),
        )

    def wrap_tool_call(
        self, request: ToolCallRequest, handler
    ) -> ToolMessage | Command:
        """Acknowledge a stubbed call; run any other call."""
        return self._stubbed(request) or handler(request)

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler
    ) -> ToolMessage | Command:
        """Acknowledge a stubbed call; run any other call."""
        return self._stubbed(request) or await handler(request)
