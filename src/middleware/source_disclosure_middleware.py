"""Disclose unavailable support knowledge-base research."""

from __future__ import annotations

import re
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
_DISCLOSURE = (
    "Support articles could not be consulted, so this answer is based on official "
    "documentation only."
)
_DISCLOSURE_PATTERN = re.compile(
    r"support articles.*(?:could not be consulted|unable to consult|unavailable|not available)",
    re.IGNORECASE | re.DOTALL,
)
_FAILURE_PATTERNS = (
    "PylonUnavailableError",
    "Pylon API returned HTTP",
    "could not be reached",
    "No articles available in the knowledge base.",
    "No articles returned from API",
    "No published articles available in the knowledge base.",
    "No content available.",
)


class SourceDisclosureMiddleware(AgentMiddleware):
    """Disclose failed support knowledge-base research in final answers."""

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Add a support research disclosure when the knowledge base failed."""
        response = await handler(request)
        response_messages = self._response_messages(response)
        if self._has_pending_tool_calls(response_messages):
            return response

        latest_human_index = self._latest_human_index(request.messages)
        if latest_human_index < 0:
            return response
        turn_messages = request.messages[latest_human_index + 1 :]
        if not self._support_leg_failed(turn_messages):
            return response

        assistant_message = self._assistant_message(response_messages)
        if assistant_message is None:
            return response
        text = self._message_text(assistant_message)
        if _DISCLOSURE_PATTERN.search(text):
            return response
        return self._replace_message(
            response, assistant_message, self._add_disclosure(text, assistant_message.content)
        )

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

    def _support_leg_failed(self, messages: list[BaseMessage]) -> bool:
        support_messages = [
            message
            for message in messages
            if isinstance(message, ToolMessage) and message.name in SUPPORT_KB_TOOLS
        ]
        if not support_messages:
            return False
        return not any(
            message.name == "get_support_article_content"
            and self._usable_article_content(message)
            for message in support_messages
        )

    def _usable_article_content(self, message: ToolMessage) -> bool:
        if getattr(message, "status", None) == "error":
            return False
        text = self._message_text(message).strip()
        if not text:
            return False
        return not self._contains_failure(text)

    def _contains_failure(self, text: str) -> bool:
        lowered = text.lower()
        return any(pattern.lower() in lowered for pattern in _FAILURE_PATTERNS) or bool(
            re.search(r"article id .* not found in knowledge base\.", lowered)
        )

    def _assistant_message(self, messages: list[BaseMessage]) -> AIMessage | None:
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                return message
        return None

    def _add_disclosure(self, text: str, content: Any) -> str | list[Any]:
        if isinstance(content, list):
            return self._add_disclosure_to_blocks(content)
        return self._insert_disclosure(text)

    def _insert_disclosure(self, text: str) -> str:
        lines = text.splitlines()
        opening_index = next(
            (
                index
                for index, line in enumerate(lines)
                if re.fullmatch(r"\s*\*\*.+\*\*\s*", line)
            ),
            None,
        )
        if opening_index is None:
            return f"{_DISCLOSURE}\n\n{text}".strip()
        lines[opening_index + 1 : opening_index + 1] = ["", _DISCLOSURE, ""]
        return "\n".join(lines)

    def _add_disclosure_to_blocks(self, content: list[Any]) -> list[Any]:
        blocks = [block.copy() if isinstance(block, dict) else block for block in content]
        for block in blocks:
            if isinstance(block, dict) and "text" in block:
                block["text"] = self._insert_disclosure(str(block["text"]))
                return blocks
        return [{"type": "text", "text": _DISCLOSURE}, *blocks]

    def _replace_message(
        self, response: ModelResponse, message: AIMessage, content: str | list[Any]
    ) -> ModelResponse:
        messages = self._response_messages(response)
        index = messages.index(message)
        messages[index] = message.model_copy(update={"content": content})
        response.result = messages
        return response

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


__all__ = ["SourceDisclosureMiddleware", "SUPPORT_KB_TOOLS"]
