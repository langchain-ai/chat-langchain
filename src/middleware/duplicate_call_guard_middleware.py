"""Suppress duplicate tool calls within a single human turn."""

import json
import threading
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import END
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

_MAX_TOOL_CALLS_PER_TURN = 32
_DUPLICATE_NOTE = (
    "This exact tool call was already made on this turn; do not repeat it."
)
_CHECK_LINKS_REFUSAL = (
    "check_links may only be called once per turn. Finalize using the links "
    "already validated."
)
_TOOL_CALL_LIMIT_NOTE = (
    "The per-turn tool-call limit was reached. Stop using tools and provide the "
    "best answer possible from the information already gathered."
)
_IN_FLIGHT = object()


@dataclass
class _TurnState:
    seen_calls: dict[tuple[str, str], str | object] = field(default_factory=dict)
    check_links_used: bool = False
    tool_call_count: int = 0


class DuplicateCallGuardMiddleware(AgentMiddleware):
    """Suppress duplicate calls and enforce the check_links turn budget."""

    def __init__(self, max_tool_calls: int = _MAX_TOOL_CALLS_PER_TURN) -> None:
        """Initialize the per-turn tool-call ceiling and shared state."""
        self.max_tool_calls = max_tool_calls
        self._turn_states: dict[str, _TurnState] = {}
        self._state_lock = threading.Lock()

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Handle a tool call with per-turn duplicate suppression."""
        tool_name = str(request.tool_call.get("name", "unknown_tool"))
        call_key = (tool_name, self._canonical_args(request.tool_call.get("args", {})))
        state_key = self._state_key(request)
        with self._state_lock:
            turn_state = self._turn_states.setdefault(state_key, _TurnState())
            if turn_state.tool_call_count >= self.max_tool_calls:
                return self._limit_command(request)
            turn_state.tool_call_count += 1
            if tool_name == "check_links" and turn_state.check_links_used:
                return self._tool_message(request, _CHECK_LINKS_REFUSAL)
            if tool_name == "check_links":
                turn_state.check_links_used = True
            cached_content = turn_state.seen_calls.get(call_key)
            if cached_content is not None:
                if cached_content is _IN_FLIGHT:
                    return self._tool_message(request, _DUPLICATE_NOTE)
                return self._tool_message(
                    request,
                    f"{_DUPLICATE_NOTE}\n{cached_content}",
                )
            turn_state.seen_calls[call_key] = _IN_FLIGHT

        try:
            result = await handler(request)
        except Exception:
            with self._state_lock:
                turn_state.seen_calls.pop(call_key, None)
            raise
        if isinstance(result, ToolMessage) and result.status == "success":
            with self._state_lock:
                turn_state.seen_calls[call_key] = self._content_text(result.content)
        else:
            with self._state_lock:
                turn_state.seen_calls.pop(call_key, None)
        return result

    def _state_key(self, request: ToolCallRequest) -> str:
        runtime = request.runtime
        config = getattr(runtime, "config", None)
        configurable = config.get("configurable", {}) if isinstance(config, Mapping) else {}
        thread_id = configurable.get("thread_id") if isinstance(configurable, Mapping) else None
        scope = f"thread:{thread_id}" if thread_id else "default"
        return f"{scope}:{self._turn_key(request.state)}"

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

    def _limit_command(self, request: ToolCallRequest) -> Command:
        return Command(
            update={
                "messages": [
                    self._tool_message(request, _TOOL_CALL_LIMIT_NOTE),
                    AIMessage(content=_TOOL_CALL_LIMIT_NOTE),
                ]
            },
            goto=END,
        )
