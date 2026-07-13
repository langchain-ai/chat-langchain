"""Cap runaway filesystem-search loops and force a pivot to docs search."""
import json
import logging

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

logger = logging.getLogger(__name__)

# Filesystem-style search tools whose repeated empty results trigger runaway loops.
GREP_TOOL_NAMES = {"grep"}
PIVOT_TOOL_NAME = "search_docs_by_lang_chain"

NO_RESULTS_MARKERS = (
    "no results found",
    "no result found",
    "no matches",
)


class DocsLoopGuardMiddleware(AgentMiddleware[AgentState]):
    """Force a pivot to docs search after repeated grep/empty-result calls."""

    def __init__(self, max_consecutive: int = 3):
        super().__init__()
        self.max_consecutive = max_consecutive

    def _consecutive_grep_calls(self, state: object) -> int:
        messages = getattr(state, "messages", None)
        if messages is None and isinstance(state, dict):
            messages = state.get("messages", [])
        count = 0
        for message in reversed(messages or []):
            if isinstance(message, AIMessage):
                continue
            if not isinstance(message, ToolMessage):
                break
            if message.name in GREP_TOOL_NAMES:
                count += 1
            else:
                break
        return count

    def _pivot_message(self, request: ToolCallRequest) -> ToolMessage:
        payload = {
            "error": "Search loop guard triggered",
            "message": (
                f"grep returned no progress after {self.max_consecutive} consecutive "
                f"calls. Stop calling grep and use {PIVOT_TOOL_NAME} instead, then "
                "synthesize a final answer."
            ),
            "next_tool": PIVOT_TOOL_NAME,
        }
        return ToolMessage(
            content=json.dumps(payload),
            name=request.tool_call.get("name", "grep"),
            tool_call_id=request.tool_call.get("id", ""),
        )

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler,
    ) -> ToolMessage | Command:
        tool_name = request.tool_call.get("name", "")
        if (
            tool_name in GREP_TOOL_NAMES
            and self._consecutive_grep_calls(request.state) >= self.max_consecutive
        ):
            logger.warning(
                "Loop guard: forcing pivot to %s after %s consecutive grep calls",
                PIVOT_TOOL_NAME,
                self.max_consecutive,
            )
            return self._pivot_message(request)
        return await handler(request)


__all__ = ["DocsLoopGuardMiddleware"]
