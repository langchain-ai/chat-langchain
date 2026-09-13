"""Sanitize leaked model response-part scaffolding."""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage

logger = logging.getLogger(__name__)

_LEADING_PART_LABEL = re.compile(r"^(?:text|end|model|assistant|tool_code|json)\n")
_TRAILING_SIGNATURE = re.compile(
    r"(?:\n(?:\d+\n)?)?[A-Za-z0-9+/]{60,}={0,2}$"
)


class ResponseSanitizerMiddleware(AgentMiddleware):
    """Remove leaked response-part scaffolding from model text blocks."""

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | AIMessage:
        """Sanitize the response returned by the next model middleware."""
        response = await handler(request)
        sanitized, changed = self._sanitize_response(response)
        if changed:
            logger.warning(
                "Sanitized response-part scaffolding from model %s",
                self._model_name(request),
            )
        return sanitized

    def _sanitize_response(
        self, response: ModelResponse | AIMessage
    ) -> tuple[ModelResponse | AIMessage, bool]:
        if isinstance(response, AIMessage):
            message, changed = self._sanitize_message(response)
            return message, changed

        changed = False
        messages: list[BaseMessage] = []
        for message in response.result:
            sanitized_message, message_changed = self._sanitize_message(message)
            messages.append(sanitized_message)
            changed = changed or message_changed
        if not changed:
            return response, False
        return response.__class__(
            result=messages,
            structured_response=response.structured_response,
        ), True

    def _sanitize_message(
        self, message: BaseMessage
    ) -> tuple[BaseMessage, bool]:
        if not isinstance(message, AIMessage) or not isinstance(message.content, list):
            return message, False

        changed = False
        content: list[Any] = []
        for block in message.content:
            if not isinstance(block, dict) or block.get("type") != "text":
                content.append(block)
                continue
            text = block.get("text")
            if not isinstance(text, str):
                content.append(block)
                continue
            sanitized_text = _LEADING_PART_LABEL.sub("", text, count=1)
            sanitized_text = _TRAILING_SIGNATURE.sub("", sanitized_text, count=1)
            if sanitized_text == text:
                content.append(block)
                continue
            content.append({**block, "text": sanitized_text})
            changed = True

        if not changed:
            return message, False
        return message.model_copy(update={"content": content}), True

    def _model_name(self, request: ModelRequest) -> str:
        model = request.model
        return str(
            getattr(model, "model_name", None)
            or getattr(model, "model", None)
            or model.__class__.__name__
        )


__all__ = ["ResponseSanitizerMiddleware"]
