"""Keep historical tool names safe for retry and fallback providers."""

import re
from collections.abc import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, ToolMessage


class ToolNameSanitizerMiddleware(AgentMiddleware):
    """Sanitize request-local history without changing checkpointed messages."""

    def _sanitize_request(self, request: ModelRequest) -> ModelRequest:
        names = {
            tool.get("function", tool).get("name")
            if isinstance(tool, dict)
            else tool.name
            for tool in request.tools
        }

        def is_valid(name: object) -> bool:
            return (
                isinstance(name, str)
                and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name) is not None
                and name in names
            )

        messages = list(request.messages)
        sanitized_ids = set()
        for index, message in enumerate(messages):
            if not isinstance(message, AIMessage):
                continue
            copied = message.model_copy(deep=True)
            for call in copied.tool_calls:
                if not is_valid(call["name"]):
                    call["args"]["_original_name"] = call["name"]
                    call["name"] = "invalid_tool_call"
                    sanitized_ids.add(call["id"])
            for invalid_call in copied.invalid_tool_calls:
                if not is_valid(invalid_call["name"]):
                    invalid_call["name"] = "invalid_tool_call"
                    sanitized_ids.add(invalid_call["id"])
            for raw_call in copied.additional_kwargs.get("tool_calls", []):
                function = raw_call.get("function", {})
                if not is_valid(function.get("name")):
                    function["name"] = "invalid_tool_call"
            function_call = copied.additional_kwargs.get("function_call")
            if function_call is not None and (
                not isinstance(function_call, dict)
                or not is_valid(function_call.get("name"))
            ):
                copied.additional_kwargs.pop("function_call")
            messages[index] = copied

        for index, message in enumerate(messages):
            if isinstance(message, ToolMessage) and (
                message.tool_call_id in sanitized_ids
                or (message.name is not None and not is_valid(message.name))
            ):
                messages[index] = message.model_copy(
                    update={"name": "invalid_tool_call"}
                )
        return request.override(messages=messages)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Pass sanitized history to the synchronous model handler."""
        return handler(self._sanitize_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Pass sanitized history to the asynchronous model handler."""
        return await handler(self._sanitize_request(request))
