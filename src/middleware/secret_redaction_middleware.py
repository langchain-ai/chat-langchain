"""Redact credential-shaped values from model responses."""

import re
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)

REDACTION_PLACEHOLDER = "YOUR_API_KEY_HERE"
SECRET_PATTERN = re.compile(
    r"(?:"
    r"sk-[A-Za-z0-9_-]{16,}|"
    r"lsv2_(?:pt|sk)_[A-Za-z0-9]{16,}|"
    r"lcl_[A-Za-z0-9]{16,}|"
    r"gh[ops]_[A-Za-z0-9]{20,}|"
    r"AKIA[0-9A-Z]{12,}|"
    r"tvly-[A-Za-z0-9]{16,}|"
    r"xoxb-[A-Za-z0-9-]{16,}|"
    r"AIza[0-9A-Za-z_-]{30,}|"
    r"pk_live_[A-Za-z0-9]{16,}|"
    r"eyJ[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+){2}"
    r")"
)


class SecretRedactionMiddleware(AgentMiddleware):
    """Redact credential-shaped values from model output."""

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Redact secrets from the model response."""
        response = await handler(request)
        return ModelResponse(
            result=[self._redact_message(message) for message in response.result],
            structured_response=response.structured_response,
        )

    @classmethod
    def _redact_message(cls, message: Any) -> Any:
        content = getattr(message, "content", None)
        redacted = cls._redact_content(content)
        if redacted is content:
            return message
        return message.model_copy(update={"content": redacted})

    @classmethod
    def _redact_content(cls, content: Any) -> Any:
        if isinstance(content, str):
            return SECRET_PATTERN.sub(REDACTION_PLACEHOLDER, content)
        if not isinstance(content, list):
            return content

        redacted: list[Any] = []
        changed = False
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    sanitized = SECRET_PATTERN.sub(REDACTION_PLACEHOLDER, text)
                    redacted.append({**block, "text": sanitized})
                    changed = changed or sanitized != text
                    continue
            redacted.append(block)
        return redacted if changed else content


__all__ = ["REDACTION_PLACEHOLDER", "SECRET_PATTERN", "SecretRedactionMiddleware"]
