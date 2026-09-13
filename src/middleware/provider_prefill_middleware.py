"""Normalize provider-specific assistant prefills before model calls."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage


def _is_google_model(model: Any) -> bool:
    llm_type = getattr(model, "_llm_type", "")
    model_module = type(model).__module__
    return llm_type in {
        "google_genai",
        "google_generative_ai",
    } or model_module.startswith("langchain_google_genai")


def _without_assistant_prefill(messages: list[Any]) -> list[Any]:
    end = len(messages)
    while end and isinstance(messages[end - 1], AIMessage):
        end -= 1
    return messages[:end] if end != len(messages) else messages


class ProviderPrefillMiddleware(AgentMiddleware):
    """Remove unsupported trailing assistant prefills from Google requests."""

    def _normalize_request(self, request: ModelRequest) -> ModelRequest:
        if not _is_google_model(request.model):
            return request
        messages = _without_assistant_prefill(request.messages)
        return (
            request
            if messages is request.messages
            else request.override(messages=messages)
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Normalize provider-specific prefills before a synchronous call."""
        return handler(self._normalize_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Normalize provider-specific prefills before an asynchronous call."""
        return await handler(self._normalize_request(request))


__all__ = ["ProviderPrefillMiddleware"]
