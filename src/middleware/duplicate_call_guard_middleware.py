"""Suppress duplicate tool calls within a single human turn."""

import json
from collections.abc import Awaitable, Callable, Mapping, MutableMapping
from typing import Any, NotRequired

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import AgentState
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

_TURN_STATE_KEY = "_chat_langchain_turn_state"
_REPAIR_BUDGET = 2
_DUPLICATE_NOTE = (
    "This exact tool call was already made on this turn; do not repeat it."
)
_CHECK_LINKS_REFUSAL = (
    "check_links may only be called once per turn. Finalize using the links "
    "already validated."
)


class _GuardState(AgentState):
    _chat_langchain_turn_state: NotRequired[dict[str, dict[str, Any]]]


class DuplicateCallGuardMiddleware(AgentMiddleware):
    """Suppress duplicate calls and enforce the check_links turn budget."""

    state_schema = _GuardState

    def __init__(self) -> None:
        """Initialize shared per-turn state."""
        self._turn_states: dict[str, dict[str, Any]] = {}

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Handle a tool call with per-turn duplicate suppression."""
        tool_name = str(request.tool_call.get("name", "unknown_tool"))
        turn_state = _shared_turn_state(request.state, self._turn_states)
        seen_calls = turn_state["seen_calls"]

        if tool_name == "check_links" and turn_state["check_links_called"]:
            return self._tool_message(request, _CHECK_LINKS_REFUSAL)

        call_key = (tool_name, self._canonical_args(request.tool_call.get("args", {})))
        cached_content = seen_calls.get(call_key)
        if cached_content is not None:
            return self._tool_message(
                request,
                f"{_DUPLICATE_NOTE}\n{cached_content}",
            )

        if tool_name == "check_links":
            turn_state["check_links_called"] = True
        seen_calls[call_key] = None

        try:
            result = await handler(request)
        except Exception:
            seen_calls.pop(call_key, None)
            raise
        if isinstance(result, ToolMessage) and result.status == "success":
            seen_calls[call_key] = self._content_text(result.content)
        else:
            seen_calls.pop(call_key, None)
        return result

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


def _shared_turn_state(
    state: Any, registry: dict[str, dict[str, Any]] | None = None
) -> dict[str, Any]:
    messages = list(state.get("messages", [])) if isinstance(state, Mapping) else []
    turn_key = _turn_key(messages)
    if not isinstance(state, MutableMapping):
        return registry.setdefault(turn_key, _new_turn_state()) if registry else _new_turn_state()
    if registry is not None:
        turn_state = registry.setdefault(turn_key, _new_turn_state())
    else:
        turn_state = None
    turns = state.setdefault(_TURN_STATE_KEY, {})
    if turn_state is not None:
        turns[turn_key] = turn_state
        return turn_state
    return turns.setdefault(turn_key, _new_turn_state())


def _new_turn_state() -> dict[str, Any]:
    return {"seen_calls": {}, "check_links_called": False, "repair_count": 0}


def _turn_key(messages: list[BaseMessage]) -> str:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, HumanMessage) or getattr(message, "type", None) == "human":
            return str(getattr(message, "id", None) or f"{index}:{message.content!r}")
    return "no-human-message"


def consume_repair_budget(state: Any, messages: list[BaseMessage]) -> bool:
    """Consume one shared repair attempt for the current human turn."""
    turn_state = _shared_turn_state_for_messages(state, messages)
    if turn_state["repair_count"] >= _REPAIR_BUDGET:
        return False
    turn_state["repair_count"] += 1
    return True


def _shared_turn_state_for_messages(
    state: Any, messages: list[BaseMessage]
) -> dict[str, Any]:
    if not isinstance(state, MutableMapping):
        return {"seen_calls": {}, "check_links_called": False, "repair_count": 0}
    turn_key = _turn_key(messages)
    turns = state.setdefault(_TURN_STATE_KEY, {})
    return turns.setdefault(
        turn_key,
        {"seen_calls": {}, "check_links_called": False, "repair_count": 0},
    )
