"""Guard model-emitted tool-call names before they enter agent state."""

from __future__ import annotations

import contextvars
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, BaseMessage

logger = logging.getLogger(__name__)

_VALID_TOOL_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}")
_FORCED_TURN: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "tool_call_name_guard_forced_turn", default=None
)


class ToolCallNameGuardMiddleware(AgentMiddleware):
    """Repair malformed model-emitted tool-call names."""

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Repair malformed tool calls before they enter agent state."""
        sanitized_request = self._sanitize_request(request)
        response = await handler(sanitized_request)
        corrected_response, regenerated = self._sanitize_response(
            response, request.tools
        )
        if regenerated and self._turn_key(request.messages) != _FORCED_TURN.get():
            _FORCED_TURN.set(self._turn_key(request.messages))
            response = await handler(request)
            corrected_response, _ = self._sanitize_response(response, request.tools)
        return corrected_response

    def _sanitize_request(self, request: ModelRequest) -> ModelRequest:
        messages, changed, _ = self._sanitize_messages(request.messages, request.tools)
        if not changed:
            return request
        return request.override(messages=messages)

    def _sanitize_response(
        self, response: ModelCallResult, tools: list[Any]
    ) -> tuple[ModelCallResult, bool]:
        response_messages = self._response_messages(response)
        messages, changed, regenerated = self._sanitize_messages(
            response_messages, tools
        )
        if not changed:
            return response, regenerated
        if isinstance(response, ModelResponse):
            return (
                ModelResponse(
                    result=messages,
                    structured_response=response.structured_response,
                ),
                regenerated,
            )
        if isinstance(response, AIMessage) and len(messages) == 1:
            return messages[0], regenerated
        return response, regenerated

    def _sanitize_messages(
        self,
        messages: list[BaseMessage],
        tools: list[Any],
    ) -> tuple[list[BaseMessage], bool, bool]:
        tool_names = self._tool_names(tools)
        sanitized: list[BaseMessage] = []
        changed = False
        regenerate = False
        for message in messages:
            if not isinstance(message, AIMessage):
                sanitized.append(message)
                continue
            tool_calls = list(message.tool_calls)
            corrected_calls = []
            message_changed = False
            for tool_call in tool_calls:
                name = tool_call.get("name")
                if isinstance(name, str) and _VALID_TOOL_NAME.fullmatch(name):
                    corrected_calls.append(tool_call)
                    continue
                logger.warning("Malformed tool-call name: %s", str(name)[:120])
                matches = [
                    tool_name for tool_name in tool_names if tool_name in str(name)
                ]
                if len(matches) == 1:
                    corrected_calls.append({**tool_call, "name": matches[0]})
                message_changed = True
            if message_changed:
                message = message.model_copy(update={"tool_calls": corrected_calls})
                changed = True
            if message_changed and not corrected_calls and not message.content:
                regenerate = True
            sanitized.append(message)
        return (sanitized if changed else messages), changed, regenerate

    def _response_messages(self, response: ModelCallResult) -> list[BaseMessage]:
        result = getattr(response, "result", None)
        if result is not None:
            return list(result)
        return [response]

    def _tool_names(self, tools: list[Any]) -> list[str]:
        names = []
        for tool in tools:
            if isinstance(tool, dict):
                name = tool.get("name")
                if name is None and isinstance(tool.get("function"), dict):
                    name = tool["function"].get("name")
            else:
                name = getattr(tool, "name", None)
            if isinstance(name, str):
                names.append(name)
        return names

    def _turn_key(self, messages: list[BaseMessage]) -> str:
        for index in range(len(messages) - 1, -1, -1):
            if getattr(messages[index], "type", None) == "human":
                human = messages[index]
                return str(getattr(human, "id", None) or f"{index}:{human.content!r}")
        return ""


__all__ = ["ToolCallNameGuardMiddleware"]
