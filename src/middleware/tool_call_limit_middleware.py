# Tool-call iteration guard: prevents runaway tool-call loops
# (e.g., agent searching the same concept with progressively broader queries).
#
# When the number of tool messages already in the conversation reaches the
# configured limit, we mutate the next ModelRequest to clear its tools list
# (and force tool_choice="none") so the model is required to respond with a
# final answer instead of issuing more tool calls.
import logging
from typing import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import ToolMessage

logger = logging.getLogger(__name__)


class ToolCallLimitMiddleware(AgentMiddleware):
    """Force the agent to respond after a maximum number of tool calls.

    This protects against pathological loops where the model keeps issuing
    new tool calls (often broader/desperate variations of the same query)
    instead of synthesizing an answer from results already in context.

    The limit is enforced at the model-request level: when the number of
    `ToolMessage`s in the request's message list reaches `max_tool_calls`,
    we strip the request's tools so the model has no choice but to emit a
    final assistant message.
    """

    def __init__(self, max_tool_calls: int = 16):
        super().__init__()
        if max_tool_calls < 1:
            raise ValueError("max_tool_calls must be >= 1")
        self.max_tool_calls = max_tool_calls

    def _count_tool_messages(self, messages: list) -> int:
        return sum(1 for m in messages if isinstance(m, ToolMessage))

    def _disable_tools(self, request: ModelRequest) -> None:
        """Mutate the request so the model cannot issue further tool calls."""
        # Clear the tools list so the model can only produce a final answer.
        if hasattr(request, "tools"):
            try:
                request.tools = []
            except Exception:
                # Some ModelRequest impls may freeze tools; best-effort.
                logger.debug("Could not clear request.tools", exc_info=True)
        # Best-effort: also force tool_choice="none" if supported.
        if hasattr(request, "tool_choice"):
            try:
                request.tool_choice = "none"
            except Exception:
                logger.debug("Could not set tool_choice='none'", exc_info=True)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        messages = getattr(request, "messages", []) or []
        tool_msg_count = self._count_tool_messages(messages)

        if tool_msg_count >= self.max_tool_calls:
            logger.warning(
                "Tool call limit reached (%d tool messages >= max=%d); "
                "disabling tools to force a final answer.",
                tool_msg_count,
                self.max_tool_calls,
            )
            self._disable_tools(request)

        return await handler(request)


__all__ = ["ToolCallLimitMiddleware"]
