"""Prevent identical tool calls from repeating within one user turn."""

from __future__ import annotations

import contextvars
import json
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

_MAX_MEMO_ENTRIES = 128
_REPEAT_NOTE = (
    "NOTE: this exact tool call was already made on this turn; the result below "
    "is unchanged. Do not call it again with the same arguments."
)
_MEMO: contextvars.ContextVar[
    tuple[str, dict[tuple[str, str], str]] | None
] = contextvars.ContextVar("repeat_tool_call_memo", default=None)


class RepeatToolCallGuardMiddleware(AgentMiddleware):
    """Reuse identical tool results within the current user turn."""

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Reuse a prior result or invoke the tool for a new signature."""
        turn_key = self._turn_key(request.state)
        if turn_key is None:
            return await handler(request)

        memo_state = _MEMO.get()
        if memo_state is None or memo_state[0] != turn_key:
            memo: dict[tuple[str, str], str] = {}
            _MEMO.set((turn_key, memo))
        else:
            memo = memo_state[1]

        signature = self._signature(request)
        if signature in memo:
            return ToolMessage(
                content=f"{_REPEAT_NOTE}\n{memo[signature]}",
                name=self._tool_name(request),
                tool_call_id=self._tool_call_id(request),
            )

        result = await handler(request)
        memo[signature] = self._result_text(result)
        if len(memo) > _MAX_MEMO_ENTRIES:
            del memo[next(iter(memo))]
        return result

    def _signature(self, request: ToolCallRequest) -> tuple[str, str]:
        tool_name = self._tool_name(request)
        args = json.dumps(
            request.tool_call.get("args", {}), sort_keys=True, default=str
        )
        return tool_name, args

    def _tool_name(self, request: ToolCallRequest) -> str:
        return request.tool_call.get("name", "unknown_tool")

    def _tool_call_id(self, request: ToolCallRequest) -> str:
        return request.tool_call.get("id", "")

    def _result_text(self, result: ToolMessage | Command) -> str:
        content: Any = getattr(result, "content", result)
        return content if isinstance(content, str) else str(content)

    def _turn_key(self, state: Any) -> str | None:
        messages = self._messages(state)
        latest_human_index = self._latest_human_index(messages)
        if latest_human_index < 0:
            return None
        human = messages[latest_human_index]
        return str(
            getattr(human, "id", None)
            or f"{latest_human_index}:{human.content!r}"
        )

    def _messages(self, state: Any) -> list[BaseMessage]:
        if isinstance(state, dict):
            return list(state.get("messages", []))
        return list(getattr(state, "messages", []))

    def _latest_human_index(self, messages: list[BaseMessage]) -> int:
        for index in range(len(messages) - 1, -1, -1):
            if getattr(messages[index], "type", None) == "human":
                return index
        return -1
