"""Suppress identical tool calls within a single human turn."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command


class DuplicateToolCallMiddleware(AgentMiddleware):
    """Return cached results when a tool call repeats within the same turn."""

    def __init__(self) -> None:
        """Initialize the per-turn result cache."""
        super().__init__()
        self._cache: dict[str, dict[tuple[str, str], str]] = {}

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Suppress an identical tool call already executed on this turn."""
        tool_name = request.tool_call.get("name", "unknown_tool")
        args = request.tool_call.get("args", {})
        signature = (tool_name, json.dumps(args, sort_keys=True, default=str))
        turn_key = self._turn_key(request.state)
        turn_cache = self._cache.setdefault(turn_key, {})
        if signature in turn_cache:
            return ToolMessage(
                content=(
                    "[duplicate call suppressed] This exact call already ran on this "
                    "turn; reuse this result and do not call it again.\n"
                    f"{turn_cache[signature]}"
                ),
                name=tool_name,
                tool_call_id=request.tool_call.get("id", ""),
            )

        result = await handler(request)
        if isinstance(result, ToolMessage):
            turn_cache[signature] = self._message_text(result)
        return result

    def _turn_key(self, state: Any) -> str:
        messages = state.get("messages", []) if state else []
        for index in range(len(messages) - 1, -1, -1):
            if getattr(messages[index], "type", None) == "human":
                human = messages[index]
                return str(getattr(human, "id", None) or f"{index}:{human.content!r}")
        return "no-human-turn"

    def _message_text(self, message: BaseMessage) -> str:
        content = getattr(message, "content", "")
        return content if isinstance(content, str) else str(content)


__all__ = ["DuplicateToolCallMiddleware"]
