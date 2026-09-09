"""Ensure substantive technical answers use fresh documentation research."""

from __future__ import annotations

import os
import re
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage

RESEARCH_TOOLS = frozenset(
    {
        "search_docs_by_lang_chain",
        "query_docs_filesystem_docs_by_lang_chain",
        "search_support_articles",
        "get_support_article_content",
    }
)
RESEARCH_GUARD_DISABLED_ENV = "DOCS_RESEARCH_GUARD_DISABLED"
_RETRY_INSTRUCTIONS = (
    "Before answering, research this question on this turn. Call "
    "search_docs_by_lang_chain and query_docs_filesystem_docs_by_lang_chain, "
    "then use the retrieved documentation to answer. Do not answer from memory."
)
_FORCED_MARKER = "docs_research_guard_forced"
_MAX_FORCED_RETRIES = 1


class DocsResearchGuardMiddleware(AgentMiddleware):
    """Force fresh documentation research before terminal technical answers."""

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Require fresh research before returning a technical answer."""
        response = await handler(request)
        if self._should_retry(request, response):
            retry_request = request.override(
                messages=[
                    *request.messages,
                    *self._mark_forced_response(self._response_messages(response)),
                ],
                system_message=self._retry_system_message(request),
            )
            return await handler(retry_request)
        return response

    def _should_retry(self, request: ModelRequest, response: ModelResponse) -> bool:
        if os.getenv(RESEARCH_GUARD_DISABLED_ENV, "").lower() in {"1", "true", "yes"}:
            return False
        messages = request.messages
        latest_human_index = self._latest_human_index(messages)
        if latest_human_index < 0:
            return False
        if (
            self._forced_retry_count(messages[latest_human_index + 1 :])
            >= _MAX_FORCED_RETRIES
        ):
            return False
        response_messages = self._response_messages(response)
        if self._has_pending_tool_calls(response_messages):
            return False
        if self._has_research_tool(messages[latest_human_index + 1 :]):
            return False
        return self._is_substantive_technical_answer(response_messages)

    def _latest_human_index(self, messages: list[BaseMessage]) -> int:
        for index in range(len(messages) - 1, -1, -1):
            if getattr(messages[index], "type", None) == "human":
                return index
        return -1

    def _response_messages(self, response: ModelResponse) -> list[BaseMessage]:
        result = getattr(response, "result", None)
        if result is not None:
            return list(result)
        return [response]

    def _mark_forced_response(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        marked = list(messages)
        for index in range(len(marked) - 1, -1, -1):
            message = marked[index]
            if isinstance(message, AIMessage):
                marked[index] = message.model_copy(
                    update={
                        "additional_kwargs": {
                            **message.additional_kwargs,
                            _FORCED_MARKER: True,
                        }
                    }
                )
                break
        return marked

    def _forced_retry_count(self, messages: list[BaseMessage]) -> int:
        return sum(
            isinstance(message, AIMessage)
            and bool(message.additional_kwargs.get(_FORCED_MARKER))
            for message in messages
        )

    def _has_pending_tool_calls(self, messages: list[BaseMessage]) -> bool:
        return any(
            isinstance(message, AIMessage) and bool(message.tool_calls)
            for message in messages
        )

    def _has_research_tool(self, messages: list[BaseMessage]) -> bool:
        return any(
            isinstance(message, ToolMessage) and message.name in RESEARCH_TOOLS
            for message in messages
        )

    def _is_substantive_technical_answer(self, messages: list[BaseMessage]) -> bool:
        text = "\n".join(self._message_text(message) for message in messages)
        if len(text.strip()) < 40:
            return False
        return bool(
            "```" in text
            or re.search(r"`[^`]+`", text)
            or re.search(r"\b[A-Z][A-Za-z0-9]+(?:\.[A-Za-z_][A-Za-z0-9_]*)?\b", text)
            or re.search(
                r"\b(?:api|class|function|method|constructor|parameter|argument|"
                r"config(?:uration)?|option|property|field|tool call|invoke|returns?)\b",
                text,
                re.IGNORECASE,
            )
        )

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
        content = f"{existing}\n\n{_RETRY_INSTRUCTIONS}".strip()
        return SystemMessage(content=content)


__all__ = ["DocsResearchGuardMiddleware", "RESEARCH_TOOLS"]
