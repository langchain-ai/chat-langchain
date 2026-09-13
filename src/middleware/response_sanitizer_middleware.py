"""Sanitize provider serialization artifacts from model responses."""

from __future__ import annotations

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
from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)

_LEADING_FRAME_PATTERN = re.compile(
    r"^(?:text|end|start|content|delta|json|thinking)\n"
)
_TRAILING_BLOB_PATTERN = re.compile(
    r"\n(?:\d\n)?[A-Za-z0-9+/]{60,}={0,2}$"
)


class ResponseSanitizerMiddleware(AgentMiddleware):
    """Remove provider serialization artifacts from AI message text."""

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Sanitize text content after the model call completes."""
        response = await handler(request)
        model_name = self._model_name(request)
        messages = self._response_messages(response)
        sanitized_messages = [
            self._sanitize_message(message, model_name) for message in messages
        ]
        if isinstance(response, AIMessage):
            return response.model_copy(update={"content": sanitized_messages[0].content})
        response.result = sanitized_messages
        return response

    def _response_messages(self, response: ModelCallResult) -> list[Any]:
        if isinstance(response, AIMessage):
            return [response]
        return list(response.result)

    def _sanitize_message(self, message: Any, model_name: str) -> Any:
        if not isinstance(message, AIMessage):
            return message
        content = message.content
        sanitized_content = self._sanitize_content(content, model_name)
        if sanitized_content == content:
            return message
        return message.model_copy(update={"content": sanitized_content})

    def _sanitize_content(self, content: Any, model_name: str) -> Any:
        if isinstance(content, str):
            return self._sanitize_text(content, model_name)
        if not isinstance(content, list):
            return content

        sanitized_blocks = []
        changed = False
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    sanitized_text = self._sanitize_text(text, model_name)
                    if sanitized_text != text:
                        block = {**block, "text": sanitized_text}
                        changed = True
            sanitized_blocks.append(block)
        return sanitized_blocks if changed else content

    def _sanitize_text(self, text: str, model_name: str) -> str:
        leading_match = _LEADING_FRAME_PATTERN.match(text)
        if leading_match:
            text = text[leading_match.end() :]
            logger.warning(
                "Sanitized model response model=%s rule=leading_frame_token",
                model_name,
            )

        trailing_match = _TRAILING_BLOB_PATTERN.search(text)
        if trailing_match:
            text = text[: trailing_match.start()]
            logger.warning(
                "Sanitized model response model=%s rule=trailing_base64_blob",
                model_name,
            )
        return text

    def _model_name(self, request: ModelRequest) -> str:
        model = request.model
        return str(
            getattr(model, "model", None)
            or getattr(model, "model_name", None)
            or type(model).__name__
        )


__all__ = ["ResponseSanitizerMiddleware"]
