"""Suppress duplicate tool calls within a single human turn."""

import asyncio
import json
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Annotated, Any
from uuid import uuid4

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.middleware.types import (
    ModelCallResult,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
)
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langgraph.config import get_config
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command
from typing_extensions import NotRequired


class _GuardState(AgentState):
    """Carry the private invocation identity across graph tasks."""

    duplicate_call_guard_invocation: NotRequired[Annotated[str, PrivateStateAttr]]


@dataclass
class _TurnState:
    """Track successful calls and the reserved link-check budget."""

    seen_calls: dict[tuple[str, str], str] = field(default_factory=dict)
    check_links_called: bool = False
    check_links_refused: bool = False


_MAX_TURN_STATES = 1000
_TURN_STATE: OrderedDict[tuple[str, str], _TurnState] = OrderedDict()
_TURN_STATE_LOCK = asyncio.Lock()
_DUPLICATE_NOTE = (
    "This exact tool call was already made on this turn; do not repeat it."
)
_CHECK_LINKS_REFUSAL = (
    "check_links may only be called once per turn. Finalize using the links "
    "already validated."
)


class DuplicateCallGuardMiddleware(AgentMiddleware):
    """Suppress duplicate calls and enforce the check_links turn budget."""

    state_schema = _GuardState

    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any]:
        """Assign a shared identity for invocations without a thread ID."""
        return {"duplicate_call_guard_invocation": str(uuid4())}

    async def abefore_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any]:
        """Assign the invocation identity before asynchronous graph steps."""
        return self.before_agent(state, runtime)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Remove check_links after a refusal so the model can finalize."""
        async with _TURN_STATE_LOCK:
            turn_state = self._turn_state(request.state, get_config())
            refused = turn_state.check_links_refused
        if refused:
            request = request.override(
                tools=[
                    tool
                    for tool in request.tools
                    if (tool.get("name") if isinstance(tool, dict) else tool.name)
                    != "check_links"
                ]
            )
        return await handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Handle a tool call with per-turn duplicate suppression."""
        tool_name = str(request.tool_call.get("name", "unknown_tool"))
        call_key = (tool_name, self._canonical_args(request.tool_call.get("args", {})))
        async with _TURN_STATE_LOCK:
            turn_state = self._turn_state(request.state, request.runtime.config)
            if tool_name == "check_links" and turn_state.check_links_called:
                turn_state.check_links_refused = True
                return self._tool_message(request, _CHECK_LINKS_REFUSAL)

            cached_content = turn_state.seen_calls.get(call_key)
            if cached_content is not None:
                return self._tool_message(
                    request,
                    f"{_DUPLICATE_NOTE}\n{cached_content}",
                )

            if tool_name == "check_links":
                turn_state.check_links_called = True

        result = await handler(request)
        if isinstance(result, ToolMessage) and result.status == "success":
            async with _TURN_STATE_LOCK:
                turn_state.seen_calls[call_key] = self._content_text(result.content)
        return result

    def _turn_state(self, state: Any, config: Mapping[str, Any]) -> _TurnState:
        turn_key = (self._execution_key(state, config), self._turn_key(state))
        current = _TURN_STATE.get(turn_key)
        if current is None:
            current = _TurnState()
            _TURN_STATE[turn_key] = current
            if len(_TURN_STATE) > _MAX_TURN_STATES:
                _TURN_STATE.popitem(last=False)
        _TURN_STATE.move_to_end(turn_key)
        return current

    def _execution_key(self, state: Any, config: Mapping[str, Any]) -> str:
        thread_id = config.get("configurable", {}).get("thread_id")
        if thread_id is not None:
            return f"thread:{thread_id}"
        return f"invocation:{state['duplicate_call_guard_invocation']}"

    def _turn_key(self, state: Any) -> str:
        messages = self._messages(state)
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if (
                isinstance(message, HumanMessage)
                or getattr(message, "type", None) == "human"
            ):
                message_id = getattr(message, "id", None)
                return (
                    f"id:{message_id}" if message_id else f"{index}:{message.content!r}"
                )
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
