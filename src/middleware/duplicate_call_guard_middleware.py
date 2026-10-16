"""Suppress duplicate tool calls within a single human turn."""

import asyncio
import hashlib
import json
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

_TurnState = tuple[str, dict[tuple[str, str], str], set[str]]
_MAX_TURN_STATES = 256
_DUPLICATE_NOTE = (
    "This exact tool call was already made on this turn; do not repeat it."
)
_CHECK_LINKS_REFUSAL = (
    "check_links may only be called once per turn. Finalize using the links "
    "already validated."
)


class DuplicateCallGuardMiddleware(AgentMiddleware):
    """Suppress duplicate calls and enforce the check_links turn budget."""

    def __init__(self) -> None:
        """Initialize bounded state shared across tools-node tasks."""
        super().__init__()
        self._turn_states: OrderedDict[str, _TurnState] = OrderedDict()
        self._lock = asyncio.Lock()

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Handle a tool call with per-turn duplicate suppression."""
        tool_name = str(request.tool_call.get("name", "unknown_tool"))
        call_key = (tool_name, self._canonical_args(request.tool_call.get("args", {})))
        async with self._lock:
            turn_state = self._turn_state(request)
            seen_calls = turn_state[1]

            if tool_name == "check_links" and "check_links" in turn_state[2]:
                return self._tool_message(request, _CHECK_LINKS_REFUSAL)

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
            async with self._lock:
                seen_calls[call_key] = self._content_text(result.content)
        return result

    def _turn_state(self, request: ToolCallRequest) -> _TurnState:
        execution_key = self._execution_key(request)
        turn_key = self._turn_key(request.state)
        current = self._turn_states.get(execution_key)
        if current is None or current[0] != turn_key:
            current = (turn_key, {}, set())
            self._turn_states[execution_key] = current
        self._turn_states.move_to_end(execution_key)
        while len(self._turn_states) > _MAX_TURN_STATES:
            self._turn_states.popitem(last=False)
        return current

    def _execution_key(self, request: ToolCallRequest) -> str:
        runtime = request.runtime
        config = getattr(runtime, "config", None)
        configurable = (
            config.get("configurable", {}) if isinstance(config, Mapping) else {}
        )
        thread_id = (
            configurable.get("thread_id") if isinstance(configurable, Mapping) else None
        )
        if thread_id is not None:
            return f"thread:{thread_id}"
        run_id = config.get("run_id") if isinstance(config, Mapping) else None
        if run_id:
            return f"run:{run_id}"
        return f"turn:{self._turn_key(request.state)}"

    def _turn_key(self, state: Any) -> str:
        messages = self._messages(state)
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if (
                isinstance(message, HumanMessage)
                or getattr(message, "type", None) == "human"
            ):
                message_id = getattr(message, "id", None)
                if message_id:
                    return f"id:{message_id}"
                content_hash = hashlib.sha256(
                    repr(message.content).encode()
                ).hexdigest()
                return f"{index}:{content_hash}"
        return "no-human-message"

    def _messages(self, state: Any) -> list[BaseMessage]:
        if isinstance(state, Mapping):
            messages = state.get("messages", [])
        else:
            messages = getattr(state, "messages", state or [])
        return list(messages)

    def _canonical_args(self, args: Any) -> str:
        return json.dumps(
            args, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )

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
