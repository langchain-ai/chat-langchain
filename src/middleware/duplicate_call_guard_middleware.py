"""Suppress duplicate tool calls within a single human turn."""

import asyncio
import json
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

_MAX_TURN_STATES = 128
_MAX_CACHED_CALLS = 256
_CallKey = tuple[str, str]
_StepCallKey = tuple[str, _CallKey]
_DUPLICATE_NOTE = (
    "This exact tool call was already made on this turn; do not repeat it."
)
_CHECK_LINKS_REFUSAL = (
    "check_links may only be called once per turn. Finalize using the links "
    "already validated."
)


@dataclass
class _TurnState:
    """Keep bounded coordination outside individual tool tasks."""

    anchor: Any
    seen_calls: OrderedDict[_StepCallKey, str] = field(default_factory=OrderedDict)
    pending: dict[_StepCallKey, asyncio.Future[str | None]] = field(
        default_factory=dict
    )
    check_links_called: bool = False


class DuplicateCallGuardMiddleware(AgentMiddleware):
    """Suppress duplicate calls and enforce the check_links turn budget."""

    def __init__(self) -> None:
        """Initialize instance-local coordination for pending calls."""
        super().__init__()
        self._turn_states: OrderedDict[tuple[Any, str, str], _TurnState] = OrderedDict()

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Handle a tool call with per-turn duplicate suppression."""
        tool_name = str(request.tool_call.get("name", "unknown_tool"))
        messages = self._messages(request.state)
        seen_calls, check_links_called = self._prior_calls(request, messages)
        turn_state = self._turn_state(request)

        if tool_name == "check_links" and (
            check_links_called or turn_state.check_links_called
        ):
            return self._tool_message(request, _CHECK_LINKS_REFUSAL)

        call_key = (tool_name, self._canonical_args(request.tool_call.get("args", {})))
        cached_content = seen_calls.get(call_key)
        if cached_content is not None:
            return self._tool_message(
                request,
                f"{_DUPLICATE_NOTE}\n{cached_content}",
            )

        step_call_key = (self._step_key(messages), call_key)
        while step_call_key in turn_state.pending:
            cached_content = await asyncio.shield(turn_state.pending[step_call_key])
            if cached_content is not None:
                return self._tool_message(
                    request, f"{_DUPLICATE_NOTE}\n{cached_content}"
                )
        cached_content = turn_state.seen_calls.get(step_call_key)
        if cached_content is not None:
            return self._tool_message(request, f"{_DUPLICATE_NOTE}\n{cached_content}")

        completed = asyncio.get_running_loop().create_future()
        cached_content = None
        turn_state.pending[step_call_key] = completed
        if tool_name == "check_links":
            turn_state.check_links_called = True

        try:
            result = await handler(request)
            if isinstance(result, ToolMessage) and result.status == "success":
                cached_content = self._content_text(result.content)
                turn_state.seen_calls[step_call_key] = cached_content
                if len(turn_state.seen_calls) > _MAX_CACHED_CALLS:
                    turn_state.seen_calls.popitem(last=False)
            return result
        finally:
            turn_state.pending.pop(step_call_key)
            completed.set_result(cached_content)
            self._prune_turn_states()

    def _prior_calls(
        self, request: ToolCallRequest, messages: list[BaseMessage]
    ) -> tuple[dict[_CallKey, str], bool]:
        for index in range(len(messages) - 1, -1, -1):
            if (
                isinstance(messages[index], HumanMessage)
                or getattr(messages[index], "type", None) == "human"
            ):
                messages = messages[index + 1 :]
                break

        calls = {}
        results = {}
        current_step = len(messages)
        current_call_id = request.tool_call.get("id", "")
        for index, message in enumerate(messages):
            if isinstance(message, AIMessage):
                for call in message.tool_calls or []:
                    if (
                        not isinstance(call, Mapping)
                        or not isinstance(call.get("id"), str)
                        or not call["id"]
                    ):
                        continue
                    calls[call["id"]] = (index, call)
                    if current_call_id and call["id"] == current_call_id:
                        current_step = index
            elif isinstance(message, ToolMessage):
                results[message.tool_call_id] = message

        seen_calls: dict[_CallKey, str] = {}
        check_links_called = False
        for call_id, (index, call) in calls.items():
            if call_id == current_call_id:
                continue
            result = results.get(call_id)
            name = str(call.get("name", "unknown_tool"))
            if name == "check_links" and (index < current_step or result is not None):
                check_links_called = True
            if result is not None and result.status == "success":
                try:
                    call_key = (name, self._canonical_args(call.get("args", {})))
                except (TypeError, ValueError):
                    continue
                seen_calls.setdefault(call_key, self._content_text(result.content))
        return seen_calls, check_links_called

    def _turn_state(self, request: ToolCallRequest) -> _TurnState:
        turn_key = (
            asyncio.get_running_loop(),
            self._execution_key(request),
            self._turn_key(request.state),
        )
        if turn_key not in self._turn_states:
            messages = self._messages(request.state)
            human = next(
                (
                    message
                    for message in reversed(messages)
                    if isinstance(message, HumanMessage)
                ),
                None,
            )
            self._turn_states[turn_key] = _TurnState(
                (messages[0] if messages else request.state, human)
            )
        self._turn_states.move_to_end(turn_key)
        self._prune_turn_states()
        return self._turn_states[turn_key]

    def _prune_turn_states(self) -> None:
        for turn_key in list(self._turn_states):
            if len(self._turn_states) <= _MAX_TURN_STATES:
                break
            if (
                turn_key != next(reversed(self._turn_states))
                and not self._turn_states[turn_key].pending
            ):
                del self._turn_states[turn_key]

    def _execution_key(self, request: ToolCallRequest) -> str:
        runtime = request.runtime
        config = getattr(runtime, "config", None)
        configurable = (
            config.get("configurable", {}) if isinstance(config, Mapping) else {}
        )
        if not isinstance(configurable, Mapping):
            configurable = {}
        thread_id = configurable.get("thread_id")
        if thread_id:
            namespace = str(configurable.get("checkpoint_ns", "")).rpartition("|")[0]
            return f"thread:{thread_id}:{namespace}"
        run_id = config.get("run_id") if isinstance(config, Mapping) else None
        if run_id:
            return f"run:{run_id}"
        messages = self._messages(request.state)
        return f"messages:{id(messages[0] if messages else request.state)}"

    def _step_key(self, messages: list[BaseMessage]) -> str:
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if isinstance(message, AIMessage):
                return f"{index}:{message.id or id(message)}"
        return "no-ai-message"

    def _turn_key(self, state: Any) -> str:
        messages = self._messages(state)
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if (
                isinstance(message, HumanMessage)
                or getattr(message, "type", None) == "human"
            ):
                return f"{index}:{getattr(message, 'id', None) or id(message)}"
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
