"""Disclose unavailable support knowledge-base research in final answers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

SUPPORT_KB_TOOLS = frozenset(
    {"search_support_articles", "get_support_article_content"}
)
DISCLOSURE = (
    "*Support articles could not be consulted, so this answer is based on official "
    "documentation only.*"
)


class SupportKBDisclosureMiddleware(AgentMiddleware):
    """Disclose support knowledge-base errors before documentation citations."""

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Add a support knowledge-base disclosure to terminal answers."""
        response = await handler(request)
        response_messages = self._response_messages(response)
        if self._has_pending_tool_calls(response_messages):
            return response

        latest_human_index = self._latest_human_index(request.messages)
        if latest_human_index < 0:
            return response
        turn_messages = request.messages[latest_human_index + 1 :]
        if not self._has_support_kb_error(turn_messages):
            return response

        terminal_message = self._terminal_message(response_messages)
        if terminal_message is None:
            return response
        text = self._message_text(terminal_message)
        if self._has_disclosure(text):
            return response

        if "Relevant docs:" in text:
            text = text.replace("Relevant docs:", f"{DISCLOSURE}\n\nRelevant docs:", 1)
        else:
            text = f"{text.rstrip()}\n\n{DISCLOSURE}"
        messages = response_messages
        messages[messages.index(terminal_message)] = terminal_message.model_copy(
            update={"content": text}
        )
        response.result = messages
        return response

    def _latest_human_index(self, messages: list[BaseMessage]) -> int:
        for index in range(len(messages) - 1, -1, -1):
            if getattr(messages[index], "type", None) == "human":
                return index
        return -1

    def _response_messages(self, response: ModelResponse) -> list[BaseMessage]:
        result = getattr(response, "result", None)
        return list(result) if result is not None else [response]

    def _has_pending_tool_calls(self, messages: list[BaseMessage]) -> bool:
        return any(
            isinstance(message, AIMessage) and bool(message.tool_calls)
            for message in messages
        )

    def _has_support_kb_error(self, messages: list[BaseMessage]) -> bool:
        return any(
            isinstance(message, ToolMessage)
            and message.name in SUPPORT_KB_TOOLS
            and message.status == "error"
            for message in messages
        )

    def _terminal_message(self, messages: list[BaseMessage]) -> AIMessage | None:
        for message in reversed(messages):
            if isinstance(message, AIMessage) and not message.tool_calls:
                return message
        return None

    def _has_disclosure(self, text: str) -> bool:
        normalized = text.lower()
        return "support articles could not be consulted" in normalized or (
            "support knowledge base" in normalized
            and "unavailable" in normalized
        )

    def _message_text(self, message: BaseMessage) -> str:
        content: Any = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        return str(content)


__all__ = ["SupportKBDisclosureMiddleware"]
