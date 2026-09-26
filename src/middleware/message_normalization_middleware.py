"""Normalize legacy AI message fields before model and checkpoint operations."""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.runtime import Runtime


class MessageNormalizationMiddleware(AgentMiddleware):
    """Remove corrupt legacy fields from parallel tool-call messages."""

    def before_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Normalize messages before the model is called."""
        return self._normalized_update(state.get("messages", []))

    def after_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Normalize messages after the model responds."""
        return self._normalized_update(state.get("messages", []))

    def _normalized_update(
        self, messages: list[BaseMessage]
    ) -> dict[str, list[BaseMessage]] | None:
        normalized_messages = [self._normalize_message(message) for message in messages]
        if normalized_messages == messages:
            return None
        return {"messages": normalized_messages}

    def _normalize_message(self, message: BaseMessage) -> BaseMessage:
        if not isinstance(message, AIMessage) or len(message.tool_calls) <= 1:
            return message

        if "function_call" not in message.additional_kwargs:
            return message

        additional_kwargs = {
            key: value
            for key, value in message.additional_kwargs.items()
            if key != "function_call"
        }
        return message.model_copy(update={"additional_kwargs": additional_kwargs})


__all__ = ["MessageNormalizationMiddleware"]
