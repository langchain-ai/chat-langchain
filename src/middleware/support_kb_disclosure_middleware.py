"""Disclose unavailable support knowledge-base retrieval."""

from __future__ import annotations

import contextvars
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage

SUPPORT_KB_TOOLS = frozenset({"search_support_articles", "get_support_article_content"})
DISCLOSURE = (
    "Support articles could not be consulted, so this answer is based on official "
    "documentation only."
)
_RETRY_INSTRUCTIONS = f"Include this sentence verbatim: {DISCLOSURE}"
_FORCED_TURN: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "support_kb_disclosure_forced_turn", default=None
)


class SupportKbDisclosureMiddleware(AgentMiddleware):
    """Ensure answers disclose failed support knowledge-base retrieval."""

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Disclose failed support retrieval in substantive final answers."""
        response = await handler(request)
        if self._has_pending_tool_calls(self._response_messages(response)):
            return response

        latest_human_index = self._latest_human_index(request.messages)
        if latest_human_index < 0:
            return response
        turn_messages = request.messages[latest_human_index + 1 :]
        if not self._has_failed_support_tool(turn_messages):
            return response
        final_text = self._final_ai_text(self._response_messages(response))
        if len(final_text.strip()) < 120 or DISCLOSURE in final_text:
            return response

        turn_key = self._turn_key(request.messages)
        if turn_key != _FORCED_TURN.get():
            _FORCED_TURN.set(turn_key)
            retry_request = request.override(
                messages=[*request.messages, *self._response_messages(response)],
                system_message=self._retry_system_message(request),
            )
            retry_response = await handler(retry_request)
            if self._has_pending_tool_calls(self._response_messages(retry_response)):
                return retry_response
            if DISCLOSURE in self._final_ai_text(
                self._response_messages(retry_response)
            ):
                return retry_response
            return self._append_disclosure(retry_response)

        return self._append_disclosure(response)

    def _latest_human_index(self, messages: list[BaseMessage]) -> int:
        for index in range(len(messages) - 1, -1, -1):
            if getattr(messages[index], "type", None) == "human":
                return index
        return -1

    def _turn_key(self, messages: list[BaseMessage]) -> str:
        index = self._latest_human_index(messages)
        human = messages[index]
        return str(getattr(human, "id", None) or f"{index}:{human.content!r}")

    def _response_messages(self, response: ModelResponse) -> list[BaseMessage]:
        result = getattr(response, "result", None)
        return list(result) if result is not None else [response]

    def _has_pending_tool_calls(self, messages: list[BaseMessage]) -> bool:
        return any(
            isinstance(message, AIMessage) and bool(message.tool_calls)
            for message in messages
        )

    def _has_failed_support_tool(self, messages: list[BaseMessage]) -> bool:
        return any(
            isinstance(message, ToolMessage)
            and message.name in SUPPORT_KB_TOOLS
            and self._tool_message_failed(message)
            for message in messages
        )

    def _tool_message_failed(self, message: ToolMessage) -> bool:
        text = self._message_text(message)
        return (
            getattr(message, "status", None) == "error"
            or bool(getattr(message, "error", False))
            or "PylonUnavailableError" in text
            or "Pylon API returned HTTP" in text
        )

    def _final_ai_text(self, messages: list[BaseMessage]) -> str:
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                return self._message_text(message)
        return ""

    def _append_disclosure(self, response: ModelResponse) -> ModelResponse:
        messages = self._response_messages(response)
        for index in range(len(messages) - 1, -1, -1):
            if isinstance(messages[index], AIMessage):
                message = messages[index].model_copy(
                    update={
                        "content": (
                            f"{self._message_text(messages[index]).rstrip()}\n\n{DISCLOSURE}"
                        )
                    }
                )
                messages[index] = message
                response.result = messages
                return response
        return response

    def _message_text(self, message: BaseMessage) -> str:
        content: Any = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        return str(content)

    def _retry_system_message(self, request: ModelRequest) -> SystemMessage:
        existing = request.system_message.text if request.system_message else ""
        return SystemMessage(content=f"{existing}\n\n{_RETRY_INSTRUCTIONS}".strip())


__all__ = ["DISCLOSURE", "SupportKbDisclosureMiddleware"]
