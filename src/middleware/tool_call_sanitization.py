"""Sanitize tool-call names before checkpointing and model dispatch."""

from __future__ import annotations

import re
from typing import Any, Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, AnyMessage
from langgraph.runtime import Runtime

_VALID_TOOL_NAME = re.compile(r"[^a-zA-Z0-9_-]")


def sanitize_tool_name(name: Any) -> str:
    """Normalize a tool name to the provider-compatible character set."""
    sanitized = _VALID_TOOL_NAME.sub("_", str(name or ""))
    return sanitized or "_"


def _sanitize_ai_message(message: AIMessage) -> AIMessage:
    tool_calls = []
    changed = False
    for tool_call in message.tool_calls:
        sanitized_name = sanitize_tool_name(tool_call.get("name"))
        sanitized_tool_call = tool_call
        if sanitized_name != tool_call.get("name"):
            sanitized_tool_call = {**tool_call, "name": sanitized_name}
            changed = True
        tool_calls.append(sanitized_tool_call)

    additional_kwargs = message.additional_kwargs
    function_call = additional_kwargs.get("function_call")
    if isinstance(function_call, dict) and "name" in function_call:
        sanitized_name = sanitize_tool_name(function_call["name"])
        if sanitized_name != function_call["name"]:
            additional_kwargs = {
                **additional_kwargs,
                "function_call": {**function_call, "name": sanitized_name},
            }
            changed = True

    if not changed:
        return message
    return message.model_copy(
        update={"tool_calls": tool_calls, "additional_kwargs": additional_kwargs}
    )


def sanitize_messages(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Normalize tool-call names in a message list."""
    sanitized_messages = [
        _sanitize_ai_message(message) if isinstance(message, AIMessage) else message
        for message in messages
    ]
    return sanitized_messages


def _changed_messages(messages: list[AnyMessage]) -> list[AnyMessage] | None:
    sanitized_messages = sanitize_messages(messages)
    changed = [
        sanitized
        for original, sanitized in zip(messages, sanitized_messages)
        if sanitized is not original
    ]
    return changed or None


class ToolCallSanitizationMiddleware(AgentMiddleware):
    """Keep tool-call names safe in state and outbound model requests."""

    def before_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Sanitize messages before a model call."""
        return self._state_update(state)

    def after_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Sanitize model output before checkpointing."""
        return self._state_update(state)

    def _state_update(self, state: AgentState) -> dict[str, Any] | None:
        messages = _changed_messages(state.get("messages", []))
        return {"messages": messages} if messages else None

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Sanitize messages before synchronous model dispatch."""
        return handler(request.override(messages=sanitize_messages(request.messages)))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Sanitize messages before asynchronous model dispatch."""
        return await handler(request.override(messages=sanitize_messages(request.messages)))


__all__ = [
    "ToolCallSanitizationMiddleware",
    "sanitize_messages",
    "sanitize_tool_name",
]
