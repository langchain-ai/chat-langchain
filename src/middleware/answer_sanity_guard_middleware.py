"""Reject documentation-source fragments as terminal answers."""

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
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

_RETRY_INSTRUCTIONS = (
    "Answer the user's question in clear prose using this turn's retrieved documentation. "
    "End with a Relevant docs: footer containing only URLs that check_links reported valid."
)
_APOLOGY = (
    "I'm sorry, but I couldn't produce a reliable answer from the retrieved documentation. "
    "Please rephrase your question or try again."
)
_LOWERCASE_FRAGMENT_PATTERN = re.compile(r"^[a-z][a-z-]*\b")
_MDX_ARTIFACTS = (
    "</Tab>",
    "<Tabs",
    "</Tabs>",
    "<Callout",
    "<div className",
    "Edit this page on GitHub",
)
_TOOL_OUTPUT_FRAMING = ("--- stdout ---", "exit: 0")


class AnswerSanityGuardMiddleware(AgentMiddleware):
    """Keep terminal documentation answers in usable prose."""

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Validate a terminal answer and retry one degenerate response."""
        response = await handler(request)
        response_messages = self._response_messages(response)
        if self._has_pending_tool_calls(response_messages):
            return response

        message = self._last_ai_message(response_messages)
        if message is None or not self._is_degenerate(message):
            return response

        retry_response = await handler(
            request.override(
                messages=[
                    *request.messages,
                    HumanMessage(content=_RETRY_INSTRUCTIONS),
                ]
            )
        )
        retry_messages = self._response_messages(retry_response)
        if self._has_pending_tool_calls(retry_messages):
            return retry_response
        retry_message = self._last_ai_message(retry_messages)
        if retry_message is None or not self._is_degenerate(retry_message):
            return retry_response
        return self._replace_message_content(retry_response, retry_message, _APOLOGY)

    def _response_messages(self, response: ModelResponse) -> list[BaseMessage]:
        result = getattr(response, "result", None)
        return list(result) if result is not None else []

    def _has_pending_tool_calls(self, messages: list[BaseMessage]) -> bool:
        return any(
            isinstance(message, AIMessage) and bool(message.tool_calls)
            for message in messages
        )

    def _last_ai_message(self, messages: list[BaseMessage]) -> AIMessage | None:
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                return message
        return None

    def _is_degenerate(self, message: AIMessage) -> bool:
        text = self._message_text(message).strip()
        if any(artifact in text for artifact in _MDX_ARTIFACTS):
            return True
        if any(framing in text for framing in _TOOL_OUTPUT_FRAMING):
            return True
        if text.endswith("}-->"):
            return True
        return len(text) > 40 and bool(_LOWERCASE_FRAGMENT_PATTERN.match(text))

    def _message_text(self, message: BaseMessage) -> str:
        content: Any = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                text for part in content if (text := self._content_part_text(part))
            )
        return str(content)

    def _content_part_text(self, part: Any) -> str:
        if isinstance(part, dict):
            if part.get("type") == "text" or "text" in part:
                return str(part.get("text", ""))
            return ""
        if isinstance(part, list):
            return "\n".join(self._content_part_text(value) for value in part)
        return str(part)

    def _replace_message_content(
        self, response: ModelResponse, message: AIMessage, content: str
    ) -> ModelResponse:
        messages = list(getattr(response, "result", []) or [])
        index = messages.index(message)
        messages[index] = message.model_copy(update={"content": content})
        response.result = messages
        return response


__all__ = ["AnswerSanityGuardMiddleware"]
