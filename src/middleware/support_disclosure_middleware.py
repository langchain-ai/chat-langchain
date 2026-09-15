"""Disclose support knowledge-base outages in final answers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage

SUPPORT_TOOLS = frozenset(
    {
        "search_support_articles",
        "get_support_article_content",
    }
)
DISCLOSURE = (
    "Support articles could not be consulted, so this answer is based on official "
    "documentation only."
)
_RETRY_INSTRUCTIONS = (
    "State clearly that support articles could not be consulted and that this answer "
    "is based on official documentation only."
)


class SupportDisclosureMiddleware(AgentMiddleware):
    """Ensure support knowledge-base outages are disclosed."""

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Validate and repair support knowledge-base outage disclosures."""
        response = await handler(request)
        response_messages = self._response_messages(response)
        if self._has_pending_tool_calls(response_messages):
            return response

        latest_human_index = self._latest_human_index(request.messages)
        if latest_human_index < 0:
            return response
        turn_messages = request.messages[latest_human_index + 1 :]
        if not self._has_support_error(turn_messages):
            return response

        final_message = self._final_ai_message(response_messages)
        if final_message is None or self._has_disclosure(
            self._message_text(final_message)
        ):
            return response

        retry_request = request.override(
            messages=[*request.messages, *response_messages],
            system_message=self._retry_system_message(request),
        )
        repaired_response = await handler(retry_request)
        repaired_messages = self._response_messages(repaired_response)
        if self._has_pending_tool_calls(repaired_messages):
            return repaired_response
        repaired_message = self._final_ai_message(repaired_messages)
        if repaired_message is not None and not self._has_disclosure(
            self._message_text(repaired_message)
        ):
            repaired_response.result = self._insert_disclosure(
                repaired_messages, repaired_message
            )
        return repaired_response

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

    def _has_support_error(self, messages: list[BaseMessage]) -> bool:
        return any(
            isinstance(message, ToolMessage)
            and message.name in SUPPORT_TOOLS
            and (
                message.status == "error"
                or "PylonUnavailableError" in self._message_text(message)
                or "Pylon API returned HTTP 401" in self._message_text(message)
                or "Pylon API returned HTTP 403" in self._message_text(message)
            )
            for message in messages
        )

    def _final_ai_message(self, messages: list[BaseMessage]) -> AIMessage | None:
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                return message
        return None

    def _has_disclosure(self, text: str) -> bool:
        lowered = text.lower()
        return (
            "support articles could not be consulted" in lowered
            or "based on official documentation only" in lowered
        )

    def _insert_disclosure(
        self, messages: list[BaseMessage], message: AIMessage
    ) -> list[BaseMessage]:
        updated_messages = list(messages)
        index = updated_messages.index(message)
        text = self._message_text(message).rstrip()
        updated_messages[index] = message.model_copy(
            update={"content": f"{text}\n\n{DISCLOSURE}"}
        )
        return updated_messages

    def _message_text(self, message: BaseMessage) -> str:
        content: Any = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(self._content_part_text(part) for part in content)
        return str(content)

    def _content_part_text(self, part: Any) -> str:
        if isinstance(part, dict):
            return "\n".join(self._content_part_text(value) for value in part.values())
        if isinstance(part, list):
            return "\n".join(self._content_part_text(value) for value in part)
        return str(part)

    def _retry_system_message(self, request: ModelRequest) -> SystemMessage:
        existing = request.system_message.text if request.system_message else ""
        return SystemMessage(content=f"{existing}\n\n{_RETRY_INSTRUCTIONS}".strip())


__all__ = ["SupportDisclosureMiddleware"]
