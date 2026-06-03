"""Break filesystem-search no-match retry loops and guard empty final messages."""
import logging
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command

logger = logging.getLogger(__name__)

# Tool-name prefixes whose shell-style "no match" returns we want to police.
# The MCP docs filesystem tools wrap ripgrep/grep/head and report no-match
# results as `exit: 1` with empty stdout; the agent has been observed looping
# 20+ times on the same paths with minor flag variants instead of pivoting.
FILESYSTEM_TOOL_PREFIXES: tuple[str, ...] = ("query_docs_filesystem",)

# Markers that identify a "no results / exit 1" tool result without us needing
# to parse the structured payload. Lowercased for case-insensitive matching.
NO_MATCH_MARKERS: tuple[str, ...] = (
    "exit: 1",
    "exit code: 1",
    '"exit_code": 1',
    "'exit_code': 1",
    "no results found",
    "no matches found",
)

DEFAULT_MAX_CONSECUTIVE_NO_MATCH = 5

PIVOT_MESSAGE = (
    "After {k} consecutive no-match results, the filesystem search has been "
    "disabled for this turn. Do NOT call query_docs_filesystem_* again on this "
    "turn. Instead, either (a) call a different retrieval tool such as "
    "search_docs_by_lang_chain or search_support_articles with a broader query, "
    "or (b) tell the user the answer is not in the available documentation and "
    "ask a clarifying question."
)

EMPTY_CONTENT_FALLBACK = (
    "I couldn't find a complete answer to your question in the available "
    "LangChain documentation or support articles. Could you clarify what you're "
    "trying to accomplish, or share any error messages or code you're working "
    "with? You can also browse the docs directly at https://docs.langchain.com."
)


def _is_filesystem_tool(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in FILESYSTEM_TOOL_PREFIXES)


def _is_no_match_content(content: Any) -> bool:
    if content is None:
        return False
    if isinstance(content, list):
        text = " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    else:
        text = str(content)
    lowered = text.lower()
    if not lowered.strip():
        return True
    return any(marker in lowered for marker in NO_MATCH_MARKERS)


def _ai_message_text(message: AIMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                value = part.get("text") or part.get("content")
                if isinstance(value, str):
                    parts.append(value)
        return "".join(parts)
    return ""


class NoMatchLoopGuardMiddleware(AgentMiddleware[AgentState]):
    """Stop filesystem no-match loops and guarantee a non-empty final reply."""

    def __init__(
        self,
        max_consecutive_no_match: int = DEFAULT_MAX_CONSECUTIVE_NO_MATCH,
        tool_prefixes: tuple[str, ...] = FILESYSTEM_TOOL_PREFIXES,
        fallback_message: str = EMPTY_CONTENT_FALLBACK,
    ):
        """Initialize the middleware with thresholds and fallback text."""
        super().__init__()
        self.max_consecutive_no_match = max_consecutive_no_match
        self.tool_prefixes = tool_prefixes
        self.fallback_message = fallback_message

    def _is_tracked(self, name: str) -> bool:
        return any(name.startswith(prefix) for prefix in self.tool_prefixes)

    def _count_consecutive_no_match(self, state: Any, tool_name: str) -> int:
        messages = []
        if isinstance(state, dict):
            messages = state.get("messages", []) or []
        else:
            messages = getattr(state, "messages", []) or []

        count = 0
        for message in reversed(messages):
            if not isinstance(message, ToolMessage):
                if isinstance(message, AIMessage):
                    continue
                break
            msg_name = getattr(message, "name", "") or ""
            if not self._is_tracked(msg_name):
                continue
            if _is_no_match_content(message.content):
                count += 1
            else:
                break
        return count

    def _pivot_tool_message(self, request: ToolCallRequest, k: int) -> ToolMessage:
        return ToolMessage(
            content=PIVOT_MESSAGE.format(k=k),
            name=request.tool_call.get("name", "unknown_tool"),
            tool_call_id=request.tool_call.get("id", ""),
            status="error",
        )

    def _maybe_short_circuit(
        self, request: ToolCallRequest
    ) -> ToolMessage | None:
        tool_name = request.tool_call.get("name", "")
        if not self._is_tracked(tool_name):
            return None
        prior = self._count_consecutive_no_match(request.state, tool_name)
        if prior >= self.max_consecutive_no_match:
            logger.warning(
                "Short-circuiting %s after %s consecutive no-match results",
                tool_name,
                prior,
            )
            return self._pivot_tool_message(request, prior)
        return None

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler,
    ) -> ToolMessage | Command:
        """Short-circuit filesystem tool calls after too many no-match results."""
        short_circuit = self._maybe_short_circuit(request)
        if short_circuit is not None:
            return short_circuit
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler,
    ) -> ToolMessage | Command:
        """Async variant of wrap_tool_call."""
        short_circuit = self._maybe_short_circuit(request)
        if short_circuit is not None:
            return short_circuit
        return await handler(request)

    def _empty_ai_fallback(self, state: Any) -> dict[str, Any] | None:
        messages = []
        if isinstance(state, dict):
            messages = state.get("messages", []) or []
        else:
            messages = getattr(state, "messages", []) or []
        if not messages:
            return None
        last = messages[-1]
        if not isinstance(last, AIMessage):
            return None
        if last.tool_calls:
            return None
        if _ai_message_text(last).strip():
            return None
        logger.warning(
            "Model emitted empty AIMessage with no tool calls; substituting fallback"
        )
        replacement = AIMessage(
            id=last.id,
            content=self.fallback_message,
            additional_kwargs=last.additional_kwargs,
        )
        return {"messages": [replacement]}

    def after_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Replace empty final AIMessage content with a graceful fallback."""
        return self._empty_ai_fallback(state)

    async def aafter_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Async variant of after_model."""
        return self._empty_ai_fallback(state)


__all__ = ["NoMatchLoopGuardMiddleware"]
