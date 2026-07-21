"""Guard the docs agent against re-issuing identical, unproductive tool calls.

Two reinforcing behaviors caused runaway trajectories on ``docs_agent``:

* ``query_docs_filesystem_docs_by_lang_chain`` returns bare ``exit: 1`` /
  empty-stdout payloads that the model did not reliably read as a
  "did-not-find" signal, so it retried the same command many times.
* Nothing stopped the model from re-issuing byte-identical tool calls.

This middleware wraps tool execution to (a) rewrite empty / non-zero-exit
results into an explicit ``NOT_FOUND`` marker, and (b) short-circuit
byte-identical repeat calls with a ``DUPLICATE_CALL`` marker. Duplicate
detection reads the current thread's message history, so the window is
per-thread/per-trace by construction and unrelated sessions never interfere.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

logger = logging.getLogger(__name__)

#: Tools whose repeated / empty results drove the observed looping behavior.
GUARDED_TOOLS = frozenset(
    {
        "query_docs_filesystem_docs_by_lang_chain",
        "search_docs_by_lang_chain",
    }
)

#: How many recent tool calls to consider when detecting duplicates.
DEDUP_WINDOW = 10

DUPLICATE_CALL_TEMPLATE = (
    "DUPLICATE_CALL: you already ran this exact {tool_name} with these "
    "arguments. Try different arguments or switch tools."
)

NOT_FOUND_TEMPLATE = (
    "NOT_FOUND: {command}\n"
    "Try a different path or pattern, or use search_docs_by_lang_chain for "
    "semantic search."
)


class ToolLoopGuardMiddleware(AgentMiddleware[AgentState]):
    """Dedup repeated tool calls and mark empty/failed doc lookups explicitly."""

    def __init__(self, dedup_window: int = DEDUP_WINDOW):
        """Configure the per-thread duplicate-detection window."""
        super().__init__()
        self.dedup_window = dedup_window

    def _tool_name(self, request: ToolCallRequest) -> str:
        return request.tool_call.get("name", "unknown_tool")

    def _tool_call_id(self, request: ToolCallRequest) -> str:
        return request.tool_call.get("id", "")

    def _call_key(self, tool_name: str, args: Any) -> str:
        return f"{tool_name}::{json.dumps(args, sort_keys=True, default=str)}"

    def _tool_message(self, request: ToolCallRequest, content: str) -> ToolMessage:
        return ToolMessage(
            content=content,
            name=self._tool_name(request),
            tool_call_id=self._tool_call_id(request),
        )

    def _recent_call_keys(self, request: ToolCallRequest) -> set[str]:
        """Keys of the last ``dedup_window`` tool calls on this thread."""
        state = getattr(request, "state", None) or {}
        messages = state.get("messages", []) if isinstance(state, dict) else []
        keys: list[str] = []
        for message in messages:
            if isinstance(message, AIMessage):
                for call in message.tool_calls or []:
                    name = call.get("name", "")
                    if name in GUARDED_TOOLS:
                        keys.append(self._call_key(name, call.get("args", {})))
        return set(keys[-self.dedup_window :])

    def _original_command(self, request: ToolCallRequest) -> str:
        args = request.tool_call.get("args", {})
        if isinstance(args, dict):
            command = args.get("command") or args.get("query") or args.get("path")
            if command:
                return str(command)
        return json.dumps(args, sort_keys=True, default=str)

    def _is_empty_or_failed(self, content: Any) -> bool:
        text = content if isinstance(content, str) else str(content)
        stripped = text.strip()
        if not stripped:
            return True
        lowered = stripped.lower()
        return lowered == "exit: 1" or lowered.startswith("exit: 1")

    def _guard_result(
        self, request: ToolCallRequest, result: ToolMessage | Command
    ) -> ToolMessage | Command:
        if self._tool_name(request) != "query_docs_filesystem_docs_by_lang_chain":
            return result
        if not isinstance(result, ToolMessage):
            return result
        if self._is_empty_or_failed(result.content):
            marker = NOT_FOUND_TEMPLATE.format(command=self._original_command(request))
            logger.info(
                "query_docs_filesystem returned empty/failed output; marking NOT_FOUND"
            )
            return self._tool_message(request, marker)
        return result

    def _duplicate_marker(self, request: ToolCallRequest) -> ToolMessage | None:
        tool_name = self._tool_name(request)
        if tool_name not in GUARDED_TOOLS:
            return None
        key = self._call_key(tool_name, request.tool_call.get("args", {}))
        if key in self._recent_call_keys(request):
            logger.info(
                "Short-circuiting duplicate %s call with DUPLICATE_CALL marker",
                tool_name,
            )
            return self._tool_message(
                request, DUPLICATE_CALL_TEMPLATE.format(tool_name=tool_name)
            )
        return None

    def wrap_tool_call(self, request: ToolCallRequest, handler) -> ToolMessage | Command:
        """Dedup then normalize empty/failed doc results (sync path)."""
        duplicate = self._duplicate_marker(request)
        if duplicate is not None:
            return duplicate
        return self._guard_result(request, handler(request))

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler
    ) -> ToolMessage | Command:
        """Dedup then normalize empty/failed doc results (async path)."""
        duplicate = self._duplicate_marker(request)
        if duplicate is not None:
            return duplicate
        return self._guard_result(request, await handler(request))


__all__ = [
    "ToolLoopGuardMiddleware",
    "GUARDED_TOOLS",
    "DEDUP_WINDOW",
    "DUPLICATE_CALL_TEMPLATE",
    "NOT_FOUND_TEMPLATE",
]
