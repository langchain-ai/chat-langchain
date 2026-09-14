"""Sanitize model-emitted tool-call names before dispatch and persistence."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

_CONTROL_TOKEN_PATTERN = re.compile(r"<ctrl\d+>|<\|[^>]*\|>")
_PROVIDER_TOOL_NAME_PATTERN = re.compile(r"^\w{1,64}$")


class ToolCallNameMiddleware(AgentMiddleware):
    """Keep model tool calls compatible with bound-provider tool names."""

    def _bound_tool_names(self, request: ModelRequest) -> set[str]:
        names: set[str] = set()
        for tool in request.tools or []:
            if isinstance(tool, dict):
                function = tool.get("function")
                name = (
                    function.get("name")
                    if isinstance(function, dict)
                    else tool.get("name")
                )
            else:
                name = getattr(tool, "name", None)
            if isinstance(name, str):
                names.add(name)
        return names

    def _sanitize_name(self, name: Any, bound_names: set[str]) -> str | None:
        if not isinstance(name, str):
            return None
        if name in bound_names and _PROVIDER_TOOL_NAME_PATTERN.fullmatch(name):
            return name
        cleaned = _CONTROL_TOKEN_PATTERN.sub("", name).strip()
        candidates = [
            tool_name
            for tool_name in bound_names
            if cleaned == tool_name or cleaned.endswith(tool_name)
        ]
        if len(candidates) != 1:
            return None
        repaired = candidates[0]
        if not _PROVIDER_TOOL_NAME_PATTERN.fullmatch(repaired):
            return None
        return repaired

    def _sanitize_messages(
        self,
        messages: list[BaseMessage],
        bound_names: set[str],
    ) -> tuple[list[BaseMessage], bool, bool]:
        sanitized_calls: dict[int, list[dict[str, Any]]] = {}
        removed_call_ids: set[str] = set()
        changed = False
        had_unrecoverable_call = False
        for index, message in enumerate(messages):
            if not isinstance(message, AIMessage):
                continue
            calls: list[dict[str, Any]] = []
            for tool_call in message.tool_calls:
                call = dict(tool_call)
                repaired_name = self._sanitize_name(call.get("name"), bound_names)
                if repaired_name is None:
                    call_id = call.get("id")
                    if isinstance(call_id, str):
                        removed_call_ids.add(call_id)
                    changed = True
                    had_unrecoverable_call = True
                    continue
                if repaired_name != call.get("name"):
                    call["name"] = repaired_name
                    changed = True
                calls.append(call)
            for invalid_tool_call in message.invalid_tool_calls:
                call = dict(invalid_tool_call)
                repaired_name = self._sanitize_name(call.get("name"), bound_names)
                call_id = call.get("id")
                args = call.get("args")
                if repaired_name is None or not isinstance(args, dict):
                    if isinstance(call_id, str):
                        removed_call_ids.add(call_id)
                    changed = True
                    had_unrecoverable_call = True
                    continue
                call["name"] = repaired_name
                call["type"] = "tool_call"
                call.pop("error", None)
                calls.append(call)
                changed = True
            if calls != message.tool_calls or message.invalid_tool_calls:
                sanitized_calls[index] = calls
        if not changed:
            return messages, False, False

        result: list[BaseMessage] = []
        for index, message in enumerate(messages):
            if (
                isinstance(message, ToolMessage)
                and message.tool_call_id in removed_call_ids
            ):
                continue
            if index in sanitized_calls:
                result.append(
                    message.model_copy(
                        update={
                            "tool_calls": sanitized_calls[index],
                            "invalid_tool_calls": [],
                        }
                    )
                )
            else:
                result.append(message)
        return result, True, had_unrecoverable_call

    def _correction_message(self, bound_names: set[str]) -> HumanMessage:
        names = ", ".join(sorted(bound_names)) or "none"
        return HumanMessage(
            content=(
                "The previous tool call was invalid. Use only an available tool "
                f"with an exact name from this list: {names}."
            )
        )

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Sanitize requests and responses around a model call."""
        bound_names = self._bound_tool_names(request)
        clean_messages, _, _ = self._sanitize_messages(request.messages, bound_names)
        clean_request = request.override(messages=clean_messages)
        response = await handler(clean_request)
        clean_result, _, had_unrecoverable_call = self._sanitize_messages(
            response.result, bound_names
        )
        if not had_unrecoverable_call:
            if clean_result is response.result:
                return response
            return ModelResponse(
                result=clean_result,
                structured_response=response.structured_response,
            )

        retry_request = clean_request.override(
            messages=[*clean_messages, self._correction_message(bound_names)]
        )
        retry_response = await handler(retry_request)
        final_result, _, _ = self._sanitize_messages(retry_response.result, bound_names)
        return ModelResponse(
            result=final_result,
            structured_response=retry_response.structured_response,
        )


__all__ = ["ToolCallNameMiddleware"]
