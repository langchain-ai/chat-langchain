"""Suppress duplicate tool calls within a single human turn."""

import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

_MAX_TOOL_CALLS_PER_TURN = 12
_TURN_KEY = "_duplicate_call_guard_turn"
_SEEN_CALLS_KEY = "_duplicate_call_guard_seen_calls"
_BUDGET_TOOLS_KEY = "_duplicate_call_guard_budget_tools"
_CALL_COUNT_KEY = "_duplicate_call_guard_call_count"
_DUPLICATE_NOTE = (
    "This exact tool call was already made on this turn; do not repeat it."
)
_CHECK_LINKS_REFUSAL = (
    "check_links may only be called once per turn. Finalize using the links "
    "already validated."
)


class DuplicateCallGuardState(AgentState, total=False):
    """State persisted by the duplicate-call guard."""

    _duplicate_call_guard_turn: str
    _duplicate_call_guard_seen_calls: dict[str, str]
    _duplicate_call_guard_budget_tools: set[str]
    _duplicate_call_guard_call_count: int


class DuplicateCallGuardMiddleware(AgentMiddleware):
    """Suppress duplicate calls and enforce the check_links turn budget."""

    state_schema = DuplicateCallGuardState

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Handle a tool call with per-turn duplicate suppression."""
        tool_name = str(request.tool_call.get("name", "unknown_tool"))
        state = self._state_for_turn(request)
        seen_calls = state[_SEEN_CALLS_KEY]

        if tool_name == "check_links" and tool_name in state[_BUDGET_TOOLS_KEY]:
            return self._command(request, self._tool_message(request, _CHECK_LINKS_REFUSAL), state)

        call_key = self._call_key(request)
        cached_content = seen_calls.get(call_key)
        if cached_content is not None:
            return self._command(
                request,
                self._tool_message(request, f"{_DUPLICATE_NOTE}\n{cached_content}"),
                state,
            )

        if state[_CALL_COUNT_KEY] >= _MAX_TOOL_CALLS_PER_TURN:
            return self._command(
                request,
                self._tool_message(
                    request,
                    "The per-turn tool budget is exhausted. Finalize your answer now.",
                ),
                state,
            )

        state[_CALL_COUNT_KEY] += 1
        if tool_name == "check_links":
            state[_BUDGET_TOOLS_KEY].add("check_links")

        result = await handler(request)
        if isinstance(result, ToolMessage) and result.status == "success":
            seen_calls[call_key] = self._content_text(result.content)
        return self._command(request, result, state)

    def _state_for_turn(self, request: ToolCallRequest) -> dict[str, Any]:
        state = request.state if isinstance(request.state, dict) else {}
        turn_key = self._turn_key(state)
        if state.get(_TURN_KEY) != turn_key:
            state.update(
                {
                    _TURN_KEY: turn_key,
                    _SEEN_CALLS_KEY: {},
                    _BUDGET_TOOLS_KEY: set(),
                    _CALL_COUNT_KEY: 0,
                }
            )
        state.setdefault(_SEEN_CALLS_KEY, {})
        state.setdefault(_BUDGET_TOOLS_KEY, set())
        state.setdefault(_CALL_COUNT_KEY, 0)
        return state

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

    def _call_key(self, request: ToolCallRequest) -> str:
        return "\x00".join(
            (
                str(request.tool_call.get("name", "unknown_tool")),
                self._canonical_args(request.tool_call.get("args", {})),
            )
        )

    def _content_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=False, default=str)

    def _command(
        self, request: ToolCallRequest, result: ToolMessage | Command, state: dict[str, Any]
    ) -> ToolMessage | Command:
        if isinstance(result, Command):
            update = result.update if isinstance(result.update, Mapping) else {}
            return Command(
                graph=result.graph,
                update={**update, **state},
                resume=result.resume,
                goto=result.goto,
            )
        return Command(update={**state, "messages": [result]})

    def _tool_message(self, request: ToolCallRequest, content: str) -> ToolMessage:
        return ToolMessage(
            content=content,
            name=request.tool_call.get("name", "unknown_tool"),
            tool_call_id=request.tool_call.get("id", ""),
        )
