"""Model fallback middleware with completed-turn reuse and tool-call history repair.

Completed assistant turns reuse the stored answer without invoking a model.
Wraps `langchain.agents.middleware.ModelFallbackMiddleware` to ensure the
message history satisfies OpenAI's strict tool-call ordering before each
model is invoked, stripping Gemini-only metadata for other providers.
OpenAI rejects requests where an assistant message with `tool_calls` is not followed by a
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
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)

MISSING_TOOL_RESULT_PLACEHOLDER = "[tool result unavailable during model fallback]"


def sanitize_gemini_metadata(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Copy AI messages without Gemini-only function-call metadata."""
    sanitized = messages
    keys = {"function_call", "__gemini_function_call_thought_signatures__"}
    for i, message in enumerate(messages):
        if isinstance(message, AIMessage) and keys.intersection(
            message.additional_kwargs
        ):
            if sanitized is messages:
                sanitized = list(messages)
            sanitized[i] = message.model_copy(
                update={
                    "additional_kwargs": {
                        key: value
                        for key, value in message.additional_kwargs.items()
                        if key not in keys
                    }
                }
            )
    return sanitized


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
    return messages if len(repaired) == len(messages) else repaired


class RepairingModelFallbackMiddleware(ModelFallbackMiddleware):
    """Reuse completed turns and prepare tool-call history for each target model."""

    def __init__(
        self,
        first_model: str | BaseChatModel,
        *additional_models: str | BaseChatModel,
        primary_model: BaseChatModel | None = None,
    ):
        """Configure fallback models and an optional initialized primary model."""
        super().__init__(first_model, *additional_models)
        self.primary_model = primary_model

    def _model_chain(
        self, request: ModelRequest
    ) -> tuple[BaseChatModel | None, list[BaseChatModel]]:
        """Return the primary model and fallbacks to use for this request."""
        return self.primary_model, self.models

    def _on_fallback(self, request: ModelRequest) -> None:
        """Run when the primary model failed and fallbacks are about to be tried."""

    def _repaired_request(self, request: ModelRequest) -> ModelRequest:
        messages = request.messages
        if not isinstance(request.model, ChatGoogleGenerativeAI):
            messages = sanitize_gemini_metadata(messages)
        messages = repair_tool_call_ids(messages)
        if messages is request.messages:
            return request
        return request.override(messages=messages)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | AIMessage:
        """Prepare history for the primary model and each fallback in sequence."""
        if request.messages:
            last_message = request.messages[-1]
            if isinstance(last_message, AIMessage) and not last_message.tool_calls:
                return ModelResponse(result=[last_message])
        primary_model, fallback_models = self._model_chain(request)
        if primary_model is not None:
            request = request.override(model=primary_model)
        last_exception: Exception
        try:
            return handler(self._repaired_request(request))
        except Exception as e:
            last_exception = e

        if fallback_models:
            self._on_fallback(request)
        for fallback_model in fallback_models:
            repaired = self._repaired_request(request.override(model=fallback_model))
            try:
                return handler(repaired)
            except Exception as e:
                last_exception = e
                continue

        raise last_exception

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | AIMessage:
        """Prepare history for the primary model and each fallback in sequence."""
        if request.messages:
            last_message = request.messages[-1]
            if isinstance(last_message, AIMessage) and not last_message.tool_calls:
                return ModelResponse(result=[last_message])
        primary_model, fallback_models = self._model_chain(request)
        if primary_model is not None:
            request = request.override(model=primary_model)
        last_exception: Exception
        try:
            return await handler(self._repaired_request(request))
        except Exception as e:
            last_exception = e

        if fallback_models:
            self._on_fallback(request)
        for fallback_model in fallback_models:
            repaired = self._repaired_request(request.override(model=fallback_model))
            try:
                return await handler(repaired)
            except Exception as e:
                last_exception = e
                continue

        raise last_exception


__all__ = [
    "RepairingModelFallbackMiddleware",
    "repair_tool_call_ids",
    "sanitize_gemini_metadata",
    "MISSING_TOOL_RESULT_PLACEHOLDER",
]
