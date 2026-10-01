"""Suppress duplicate tool calls within a single human turn."""

import asyncio
import contextvars
import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

_TurnState = tuple[str, dict[tuple[str, str], str], set[str]]
_TURN_STATE: contextvars.ContextVar[_TurnState | None] = contextvars.ContextVar(
    "duplicate_call_guard_turn_state", default=None
)
_DUPLICATE_NOTE = (
    "This exact tool call was already made on this turn; do not repeat it."
)
_CHECK_LINKS_REFUSAL = (
    "check_links may only be called once per turn. Finalize using the links "
    "already validated."
)


class DuplicateCallGuardMiddleware(AgentMiddleware):
    """Suppress duplicate calls and enforce the check_links turn budget."""

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Handle a tool call with per-turn duplicate suppression."""
        tool_name = str(request.tool_call.get("name", "unknown_tool"))
        turn_state = self._turn_state(request)
        seen_calls = turn_state[1]

        if tool_name == "check_links" and "check_links" in turn_state[2]:
            return self._tool_message(request, _CHECK_LINKS_REFUSAL)

        call_key = (tool_name, self._canonical_args(request.tool_call.get("args", {})))
        cached_content = seen_calls.get(call_key)
        if cached_content is not None:
            return self._tool_message(
                request,
                f"{_DUPLICATE_NOTE}\n{cached_content}",
            )

        if tool_name == "check_links":
            turn_state[2].add("check_links")

        result = await handler(request)
        if isinstance(result, ToolMessage) and result.status == "success":
            seen_calls[call_key] = self._content_text(result.content)
        return result

    def _turn_state(self, request: ToolCallRequest) -> _TurnState:
        turn_key = f"{self._execution_key(request)}:{self._turn_key(request.state)}"
        current = _TURN_STATE.get()
        if current is None or current[0] != turn_key:
            current = (turn_key, {}, set())
            _TURN_STATE.set(current)
        return current

    def _execution_key(self, request: ToolCallRequest) -> str:
        runtime = request.runtime
        config = getattr(runtime, "config", None)
        run_id = config.get("run_id") if isinstance(config, Mapping) else None
        if run_id:
            return f"run:{run_id}"
        task = asyncio.current_task()
        return f"task:{id(task)}"

    def _turn_key(self, state: Any) -> str:
        messages = self._messages(state)
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if isinstance(message, HumanMessage) or getattr(message, "type", None) == "human":
                return f"{index}:{getattr(message, 'id', None)}:{message.content!r}"
        return "no-human-message"

    def _messages(self, state: Any) -> list[BaseMessage]:
        if isinstance(state, Mapping):
            messages = state.get("messages", [])
        else:
            messages = getattr(state, "messages", state or [])
        return list(messages)

    def _canonical_args(self, args: Any) -> str:
        return json.dumps(args, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def _content_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=False, default=str)

    def _tool_message(self, request: ToolCallRequest, content: str) -> ToolMessage:
        return ToolMessage(
            content=content,
            name=request.tool_call.get("name", "unknown_tool"),
            tool_call_id=request.tool_call.get("id", ""),
        )
