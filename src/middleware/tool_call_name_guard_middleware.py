"""Normalize malformed tool-call names before model fallback."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage

_VALID_TOOL_NAME = re.compile(r"^[a-zA-Z0-9_-]+$")


class ToolCallNameGuardMiddleware(AgentMiddleware):
    """Repair or remove malformed tool-call names before model calls."""

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Normalize malformed tool-call names in the request history."""
        registered_names = self._registered_tool_names(request.tools)
        messages = [
            self._normalize_message(message, registered_names)
            for message in request.messages
        ]
        if messages != request.messages:
            request = request.override(messages=messages)
        return await handler(request)

    def _registered_tool_names(self, tools: list[Any] | None) -> set[str]:
        names: set[str] = set()
        for tool in tools or []:
            if isinstance(tool, Mapping):
                name = tool.get("name")
                if name is None and isinstance(tool.get("function"), Mapping):
                    name = tool["function"].get("name")
            else:
                name = getattr(tool, "name", None)
            if isinstance(name, str):
                names.add(name)
        return names

    def _normalize_message(self, message: Any, registered_names: set[str]) -> Any:
        if not isinstance(message, AIMessage) or not message.tool_calls:
            return message

        normalized_calls = []
        changed = False
        for tool_call in message.tool_calls:
            name = tool_call.get("name")
            if isinstance(name, str) and _VALID_TOOL_NAME.fullmatch(name):
                normalized_calls.append(tool_call)
                continue

            suffix = name.rsplit(":", 1)[-1] if isinstance(name, str) else ""
            if suffix in registered_names:
                normalized_calls.append({**tool_call, "name": suffix})
            changed = True

        if not changed:
            return message
        return message.model_copy(update={"tool_calls": normalized_calls})


__all__ = ["ToolCallNameGuardMiddleware"]
