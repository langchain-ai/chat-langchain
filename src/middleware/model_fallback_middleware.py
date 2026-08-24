"""Model fallback middleware with tool-call history repair.

Wraps `langchain.agents.middleware.ModelFallbackMiddleware` to ensure the
message history satisfies OpenAI's strict tool-call ordering before a
fallback model (notably `ChatOpenAI`) is invoked. OpenAI rejects requests
where an assistant message with `tool_calls` is not followed by a
`ToolMessage` for every `tool_call_id`; cross-provider fallbacks (e.g.
Gemini -> OpenAI) and history-rewriting summarization can leave dangling
tool calls that trigger `BadRequestError`.
"""

import logging
from typing import Awaitable, Callable

from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.types import (
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage

logger = logging.getLogger(__name__)

MISSING_TOOL_RESULT_PLACEHOLDER = "[tool result unavailable during model fallback]"


def repair_tool_call_ids(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Insert synthetic ToolMessages for any tool_call_id without a response."""
    repaired: list[AnyMessage] = list(messages)
    i = 0
    while i < len(repaired):
        message = repaired[i]
        tool_calls = getattr(message, "tool_calls", None) or []
        if isinstance(message, AIMessage) and tool_calls:
            expected_ids = [tc.get("id") for tc in tool_calls if tc.get("id")]
            seen_ids: set[str] = set()
            j = i + 1
            while j < len(repaired) and isinstance(repaired[j], ToolMessage):
                seen_ids.add(repaired[j].tool_call_id)
                j += 1
            for tool_call_id in expected_ids:
                if tool_call_id in seen_ids:
                    continue
                logger.warning(
                    "Inserting placeholder ToolMessage for dangling tool_call_id=%s",
                    tool_call_id,
                )
                repaired.insert(
                    j,
                    ToolMessage(
                        tool_call_id=tool_call_id,
                        content=MISSING_TOOL_RESULT_PLACEHOLDER,
                    ),
                )
                j += 1
            i = j
            continue
        i += 1
    return repaired


class RepairingModelFallbackMiddleware(ModelFallbackMiddleware):
    """ModelFallbackMiddleware that repairs tool-call history on fallback."""

    def _repaired_request(self, request: ModelRequest) -> ModelRequest:
        repaired_messages = repair_tool_call_ids(request.messages)
        if repaired_messages is request.messages or len(repaired_messages) == len(
            request.messages
        ):
            return request
        return request.override(messages=repaired_messages)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | AIMessage:
        """Try fallback models in sequence, repairing history before each fallback."""
        last_exception: Exception
        try:
            return handler(request)
        except Exception as e:
            last_exception = e

        for fallback_model in self.models:
            repaired = self._repaired_request(request)
            try:
                return handler(repaired.override(model=fallback_model))
            except Exception as e:
                last_exception = e
                continue

        raise last_exception

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | AIMessage:
        """Try fallback models in sequence, repairing history before each fallback."""
        last_exception: Exception
        try:
            return await handler(request)
        except Exception as e:
            last_exception = e

        for fallback_model in self.models:
            repaired = self._repaired_request(request)
            try:
                return await handler(repaired.override(model=fallback_model))
            except Exception as e:
                last_exception = e
                continue

        raise last_exception


__all__ = [
    "RepairingModelFallbackMiddleware",
    "repair_tool_call_ids",
    "MISSING_TOOL_RESULT_PLACEHOLDER",
]
