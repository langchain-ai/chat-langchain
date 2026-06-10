"""Normalize cross-provider AIMessage content blocks before model invocation."""

import logging
from typing import Any, Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)

# Keys that OpenAI's Chat Completions API accepts on a content text block.
# Google emits extra keys like ``extras`` and ``index`` which OpenAI rejects.
_OPENAI_TEXT_BLOCK_KEYS = {"type", "text"}


def _normalize_block(block: Any) -> Any:
    if isinstance(block, str):
        return {"type": "text", "text": block}
    if isinstance(block, dict) and block.get("type") == "text":
        return {k: v for k, v in block.items() if k in _OPENAI_TEXT_BLOCK_KEYS}
    return block


def normalize_message_content(messages: list) -> list:
    """Coerce list-typed AIMessage.content blocks into OpenAI-compatible objects."""
    normalized = []
    for m in messages:
        if isinstance(m, AIMessage) and isinstance(m.content, list):
            new_content = [_normalize_block(block) for block in m.content]
            if new_content != m.content:
                m = m.model_copy(update={"content": new_content})
        normalized.append(m)
    return normalized


class ContentNormalizationMiddleware(AgentMiddleware):
    """Rewrites AIMessage content blocks so cross-provider history is portable."""

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Normalize message content, then defer to the next handler."""
        request.messages = normalize_message_content(request.messages)
        return await handler(request)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Normalize message content, then defer to the next handler."""
        request.messages = normalize_message_content(request.messages)
        return handler(request)


__all__ = ["ContentNormalizationMiddleware", "normalize_message_content"]
