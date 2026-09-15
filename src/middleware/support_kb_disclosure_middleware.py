"""Require disclosure when support knowledge base research is unavailable."""

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
_DISCLOSURE_INSTRUCTIONS = (
    "The support knowledge base could not be consulted on this turn. "
    "Before answering, clearly disclose that limitation to the user and do not "
    "present the response as a complete support-documentation answer."
)
_FORCED_TURN: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "support_kb_disclosure_forced_turn", default=None
)


class SupportKBDisclosureMiddleware(AgentMiddleware):
    """Require disclosure when every support knowledge base call errors."""

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Retry once when support knowledge base failure is not disclosed."""
        response = await handler(request)
        if self._should_retry(request, response):
            _FORCED_TURN.set(self._turn_key(request.messages))
            retry_request = request.override(
                messages=[*request.messages, *self._response_messages(response)],
                system_message=self._retry_system_message(request),
            )
            return await handler(retry_request)
        return response

    def _should_retry(self, request: ModelRequest, response: ModelResponse) -> bool:
        latest_human_index = self._latest_human_index(request.messages)
        if latest_human_index < 0:
            return False
        if self._turn_key(request.messages) == _FORCED_TURN.get():
            return False
        tool_messages = [
            message
            for message in request.messages[latest_human_index + 1 :]
            if isinstance(message, ToolMessage) and message.name in SUPPORT_KB_TOOLS
        ]
        if not tool_messages or not all(
            message.status == "error" for message in tool_messages
        ):
            return False
        response_messages = self._response_messages(response)
        if any(
            isinstance(message, AIMessage) and bool(message.tool_calls)
            for message in response_messages
        ):
            return False
        text = "\n".join(
            self._message_text(message) for message in response_messages
        ).lower()
        return not (
            "support knowledge base" in text
            and any(
                marker in text
                for marker in (
                    "could not",
                    "couldn't",
                    "unable",
                    "unavailable",
                    "not consulted",
                )
            )
        )

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
        if result is not None:
            return list(result)
        return [response]

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
        return SystemMessage(
            content=f"{existing}\n\n{_DISCLOSURE_INSTRUCTIONS}".strip()
        )


__all__ = ["SUPPORT_KB_TOOLS", "SupportKBDisclosureMiddleware"]
