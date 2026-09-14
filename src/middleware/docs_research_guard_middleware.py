"""Ensure substantive technical answers use fresh documentation research."""

from __future__ import annotations

import contextvars
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

SEARCH_TOOLS = frozenset(
    {
        "search_docs_by_lang_chain",
        "search_support_articles",
    }
)
READ_TOOLS = frozenset(
    {
        "query_docs_filesystem_docs_by_lang_chain",
        "get_support_article_content",
        "read_file",
        "fetch_langchain_pricing",
    }
)
RESEARCH_TOOLS = SEARCH_TOOLS | READ_TOOLS
RESEARCH_GUARD_DISABLED_ENV = "DOCS_RESEARCH_GUARD_DISABLED"
_RETRY_INSTRUCTIONS = (
    "Before answering, research this question on this turn. Call "
    "search_docs_by_lang_chain and query_docs_filesystem_docs_by_lang_chain, "
    "then use the retrieved documentation to answer. Do not answer from memory."
)
_DISCLOSURE = (
    "Documentation could not be consulted on this turn, so the following answer "
    "may contain unverified information."
)
_MAX_FORCED_ATTEMPTS = 2
_DOCS_URL_PATTERN = re.compile(r"https://docs\.langchain\.com/[^\s<>\]\)\"']+")
_CODE_BLOCK_PATTERN = re.compile(r"```.*?(?:```|$)", re.DOTALL)
_LARGE_RESULT_POINTER_PATTERN = re.compile(r"^/large_tool_results/[^\s]+$")
_FORCED_TURN: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "docs_research_guard_forced_turn", default=None
)
_FORCED_ATTEMPTS: contextvars.ContextVar[dict[str, int]] = contextvars.ContextVar(
    "docs_research_guard_forced_attempts", default={}
)


class DocsResearchGuardMiddleware(AgentMiddleware):
    """Force fresh documentation research before terminal technical answers."""

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Require fresh research before returning a technical answer."""
        response = await handler(request)
        if not self._should_retry(request, response):
            self._clear_attempts(self._turn_key(request.messages))
            return response

        turn_key = self._turn_key(request.messages)
        while self._attempt_count(turn_key) < _MAX_FORCED_ATTEMPTS:
            self._record_attempt(turn_key)
            retry_request = request.override(
                messages=[*request.messages, *self._response_messages(response)],
                system_message=self._retry_system_message(request),
                tool_choice={
                    "type": "function",
                    "function": {"name": "search_docs_by_lang_chain"},
                },
            )
            response = await handler(retry_request)
            if self._has_pending_tool_calls(self._response_messages(response)):
                return response
            if self._has_research_tool(
                self._turn_messages(request.messages, self._response_messages(response))
            ):
                self._clear_attempts(turn_key)
                return response
            if not self._is_substantive_technical_answer(
                self._response_messages(response)
            ):
                self._clear_attempts(turn_key)
                return response

        self._clear_attempts(turn_key)
        return self._sanitize_response(request, response)

    def _should_retry(self, request: ModelRequest, response: ModelResponse) -> bool:
        if os.getenv(RESEARCH_GUARD_DISABLED_ENV, "").lower() in {"1", "true", "yes"}:
            return False
        messages = request.messages
        latest_human_index = self._latest_human_index(messages)
        if latest_human_index < 0:
            return False
        response_messages = self._response_messages(response)
        if self._has_pending_tool_calls(response_messages):
            return False
        current_turn = self._turn_messages(messages, response_messages)
        if self._has_research_tool(current_turn):
            return False
        return self._is_substantive_technical_answer(response_messages)

    def _latest_human_index(self, messages: list[BaseMessage]) -> int:
        for index in range(len(messages) - 1, -1, -1):
            if getattr(messages[index], "type", None) == "human":
                return index
        return -1

    def _turn_key(self, messages: list[BaseMessage]) -> str:
        index = self._latest_human_index(messages)
        human = messages[index]
        return str(getattr(human, "id", None) or f"{index}:{human.content!r}")

    def _turn_messages(
        self, request_messages: list[BaseMessage], response_messages: list[BaseMessage]
    ) -> list[BaseMessage]:
        latest_human_index = self._latest_human_index(request_messages)
        return [
            *request_messages[latest_human_index + 1 :],
            *response_messages,
        ]

    def _response_messages(self, response: ModelResponse) -> list[BaseMessage]:
        result = getattr(response, "result", None)
        if result is not None:
            return list(result)
        return [response]

    def _has_pending_tool_calls(self, messages: list[BaseMessage]) -> bool:
        return any(
            isinstance(message, AIMessage) and bool(message.tool_calls)
            for message in messages
        )

    def _has_research_tool(self, messages: list[BaseMessage]) -> bool:
        tool_messages = [
            message
            for message in messages
            if isinstance(message, ToolMessage)
            and message.name in RESEARCH_TOOLS
            and self._has_usable_content(message)
        ]
        for index, message in enumerate(tool_messages):
            if not self._is_large_result_pointer(message):
                return True
            if any(
                later.name == "read_file" and not self._is_large_result_pointer(later)
                for later in tool_messages[index + 1 :]
            ):
                return True
        return False

    def _has_usable_content(self, message: ToolMessage) -> bool:
        return message.status not in {"error", "failure", "failed"} and bool(
            self._message_text(message).strip()
        )

    def _is_large_result_pointer(self, message: ToolMessage) -> bool:
        return bool(
            _LARGE_RESULT_POINTER_PATTERN.fullmatch(self._message_text(message).strip())
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

    def _attempt_count(self, turn_key: str) -> int:
        return _FORCED_ATTEMPTS.get().get(turn_key, 0)

    def _record_attempt(self, turn_key: str) -> None:
        attempts = dict(_FORCED_ATTEMPTS.get())
        attempts[turn_key] = attempts.get(turn_key, 0) + 1
        _FORCED_ATTEMPTS.set(attempts)
        _FORCED_TURN.set(turn_key)

    def _clear_attempts(self, turn_key: str) -> None:
        attempts = dict(_FORCED_ATTEMPTS.get())
        attempts.pop(turn_key, None)
        _FORCED_ATTEMPTS.set(attempts)
        if _FORCED_TURN.get() == turn_key:
            _FORCED_TURN.set(None)

    def _sanitize_response(
        self, request: ModelRequest, response: ModelResponse
    ) -> ModelResponse:
        current_turn = self._turn_messages(request.messages, [])
        grounded_urls = {
            url.rstrip(".,;:")
            for message in current_turn
            if isinstance(message, ToolMessage) and message.name in RESEARCH_TOOLS
            for url in _DOCS_URL_PATTERN.findall(self._message_text(message))
        }
        messages = list(response.result)
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if not isinstance(message, AIMessage):
                continue
            text = _CODE_BLOCK_PATTERN.sub("", self._message_text(message))
            text = _DOCS_URL_PATTERN.sub(
                lambda match: (
                    match.group(0)
                    if match.group(0).rstrip(".,;:") in grounded_urls
                    else ""
                ),
                text,
            )
            messages[index] = message.model_copy(
                update={"content": f"{_DISCLOSURE}\n\n{text.strip()}"}
            )
            break
        return ModelResponse(
            result=messages,
            structured_response=response.structured_response,
        )


__all__ = [
    "DocsResearchGuardMiddleware",
    "READ_TOOLS",
    "RESEARCH_TOOLS",
    "SEARCH_TOOLS",
]
