"""Ensure substantive technical answers use fresh documentation research."""

from __future__ import annotations

import contextvars
import json
import os
import re
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
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
RETRIEVAL_TOOLS = frozenset(
    {
        "search_docs_by_lang_chain",
        "query_docs_filesystem_docs_by_lang_chain",
    }
)
RESEARCH_GUARD_DISABLED_ENV = "DOCS_RESEARCH_GUARD_DISABLED"
RETRIEVAL_RESULT_CHAR_LIMIT_ENV = "DOCS_RETRIEVAL_RESULT_CHAR_LIMIT"
RETRIEVAL_TURN_CHAR_LIMIT_ENV = "DOCS_RETRIEVAL_TURN_CHAR_LIMIT"
DEFAULT_RETRIEVAL_RESULT_CHAR_LIMIT = 15_000
DEFAULT_RETRIEVAL_TURN_CHAR_LIMIT = 120_000
OMISSION_MARKER = "\n\n[retrieval content omitted]"
_RETRY_INSTRUCTIONS = (
    "Before answering, research this question on this turn. Call "
    "search_docs_by_lang_chain and query_docs_filesystem_docs_by_lang_chain, "
    "then use the retrieved documentation to answer. Do not answer from memory."
)
_FORCED_TURN: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "docs_research_guard_forced_turn", default=None
)


class DocsResearchGuardMiddleware(AgentMiddleware):
    """Force fresh documentation research before terminal technical answers."""

    def __init__(
        self,
        retrieval_result_char_limit: int | None = None,
        retrieval_turn_char_limit: int | None = None,
    ) -> None:
        """Configure per-result and per-turn retrieval limits."""
        self.retrieval_result_char_limit = retrieval_result_char_limit or int(
            os.getenv(
                RETRIEVAL_RESULT_CHAR_LIMIT_ENV,
                str(DEFAULT_RETRIEVAL_RESULT_CHAR_LIMIT),
            )
        )
        self.retrieval_turn_char_limit = retrieval_turn_char_limit or int(
            os.getenv(
                RETRIEVAL_TURN_CHAR_LIMIT_ENV,
                str(DEFAULT_RETRIEVAL_TURN_CHAR_LIMIT),
            )
        )

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage]],
    ) -> ToolMessage:
        """Bound successful docs retrieval results before they enter context."""
        result = await handler(request)
        if not isinstance(result, ToolMessage):
            return result
        if result.name not in RETRIEVAL_TOOLS or result.status == "error":
            return result
        return self._replace_content(
            result,
            self._bound_retrieval_content(result.name, result.content),
        )

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Bound retrieval context before each model call."""
        bounded_request = self._bound_request(request)
        response = await handler(bounded_request)
        if self._should_retry(bounded_request, response):
            turn_key = self._turn_key(bounded_request.messages)
            _FORCED_TURN.set(turn_key)
            retry_request = bounded_request.override(
                messages=[
                    *bounded_request.messages,
                    *self._response_messages(response),
                ],
                system_message=self._retry_system_message(bounded_request),
            )
            return await handler(self._bound_request(retry_request))
        return response

    def _bound_request(self, request: ModelRequest) -> ModelRequest:
        messages = self._bound_messages(request.messages)
        if messages == request.messages:
            return request
        return request.override(messages=messages)

    def _bound_messages(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        bounded = [
            self._bound_message(message) if isinstance(message, ToolMessage) else message
            for message in messages
        ]
        latest_human_index = self._latest_human_index(bounded)
        if latest_human_index < 0:
            return bounded
        turn_start = latest_human_index + 1
        retrieval_indexes = [
            index
            for index in range(turn_start, len(bounded))
            if self._is_retrieval_message(bounded[index])
        ]
        total = sum(len(self._message_text(bounded[index])) for index in retrieval_indexes)
        for index in retrieval_indexes:
            if total <= self.retrieval_turn_char_limit:
                break
            message = bounded[index]
            compacted = self._replace_content(
                message,
                self._metadata_only(message.name, message.content),
            )
            total -= len(self._message_text(message)) - len(self._message_text(compacted))
            bounded[index] = compacted
        return bounded

    def _bound_message(self, message: ToolMessage) -> ToolMessage:
        if message.name not in RETRIEVAL_TOOLS or message.status == "error":
            return message
        content = self._bound_retrieval_content(message.name, message.content)
        if content == message.content:
            return message
        return self._replace_content(message, content)

    def _bound_retrieval_content(self, name: str | None, content: Any) -> Any:
        if name == "search_docs_by_lang_chain":
            return self._bound_search_content(content)
        return self._bound_filesystem_content(content)

    def _bound_search_content(self, content: Any) -> Any:
        text = self._message_text_value(content)
        if len(text) <= self.retrieval_result_char_limit:
            return content
        entries = self._search_entries(text)
        retained: list[str] = []
        remaining = len(entries)
        for entry in entries:
            marker = self._omission_marker(remaining - 1)
            candidate = "\n\n".join([*retained, entry, marker])
            if len(candidate) > self.retrieval_result_char_limit:
                break
            retained.append(entry)
            remaining -= 1
        if not retained:
            return self._truncate_text(text)
        omitted = len(entries) - len(retained)
        result = "\n\n".join(retained)
        return self._fit_with_marker(result, omitted)

    def _bound_filesystem_content(self, content: Any) -> Any:
        text = self._message_text_value(content)
        if len(text) <= self.retrieval_result_char_limit:
            return content
        metadata, body = self._filesystem_parts(text)
        if not body:
            return self._truncate_text(text)
        marker = self._omission_marker(1)
        available = self.retrieval_result_char_limit - len(metadata) - len(marker)
        if available <= 0:
            return self._truncate_text(metadata + marker)
        return metadata + body[:available] + marker

    def _search_entries(self, text: str) -> list[str]:
        try:
            parsed = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            parsed = None
        if isinstance(parsed, list):
            return [json.dumps(entry, ensure_ascii=False) for entry in parsed]
        if isinstance(parsed, dict):
            for key in ("results", "items", "documents"):
                entries = parsed.get(key)
                if isinstance(entries, list):
                    prefix = json.dumps(
                        {**parsed, key: []}, ensure_ascii=False
                    )[:-2]
                    return [prefix + json.dumps(entry, ensure_ascii=False) for entry in entries]
        entries = re.split(r"\n\s*\n(?=(?:#{1,6}\s|[-*]\s|\d+[.)]\s|Title:))", text)
        return entries if len(entries) > 1 else [text]

    def _filesystem_parts(self, text: str) -> tuple[str, str]:
        lines = text.splitlines(keepends=True)
        metadata_lines: list[str] = []
        body_start = 0
        for index, line in enumerate(lines):
            if not line.strip():
                body_start = index + 1
                break
            if self._looks_like_metadata(line):
                metadata_lines.append(line)
                body_start = index + 1
            else:
                break
        metadata = "".join(metadata_lines)
        body = "".join(lines[body_start:]) if metadata else text
        return metadata, body

    def _looks_like_metadata(self, line: str) -> bool:
        return bool(
            re.match(
                r"\s*(?:command|status|page|path|file|url|exit code|return code)\s*:",
                line,
                re.IGNORECASE,
            )
        )

    def _metadata_only(self, name: str | None, content: Any) -> str:
        text = self._message_text_value(content)
        if name == "search_docs_by_lang_chain":
            try:
                parsed = json.loads(text)
            except (TypeError, json.JSONDecodeError):
                parsed = None
            if isinstance(parsed, list):
                metadata = []
                for entry in parsed:
                    if isinstance(entry, dict):
                        metadata.append(
                            {
                                key: entry[key]
                                for key in ("title", "name", "url", "link", "path")
                                if key in entry
                            }
                        )
                if metadata:
                    text = json.dumps(metadata, ensure_ascii=False)
            else:
                text = text.splitlines()[0][:500] if text.splitlines() else text[:500]
        else:
            metadata, _ = self._filesystem_parts(text)
            text = metadata or text[:500]
        return self._fit_with_marker(text, 0)

    def _fit_with_marker(self, text: str, omitted: int) -> str:
        marker = self._omission_marker(omitted)
        available = self.retrieval_result_char_limit - len(marker)
        if available <= 0:
            return marker[: self.retrieval_result_char_limit]
        return text[:available] + marker

    def _truncate_text(self, text: str) -> str:
        return self._fit_with_marker(text, 1)

    def _omission_marker(self, omitted: int) -> str:
        suffix = f" ({omitted} result{'s' if omitted != 1 else ''} omitted)"
        return OMISSION_MARKER + suffix

    def _replace_content(self, message: ToolMessage, content: Any) -> ToolMessage:
        return message.model_copy(update={"content": content})

    def _is_retrieval_message(self, message: BaseMessage) -> bool:
        return isinstance(message, ToolMessage) and message.name in RETRIEVAL_TOOLS

    def _message_text_value(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        return str(content)

    def _message_text(self, message: BaseMessage) -> str:
        return self._message_text_value(getattr(message, "content", ""))

    def _should_retry(self, request: ModelRequest, response: ModelResponse) -> bool:
        if os.getenv(RESEARCH_GUARD_DISABLED_ENV, "").lower() in {"1", "true", "yes"}:
            return False
        messages = request.messages
        latest_human_index = self._latest_human_index(messages)
        if latest_human_index < 0:
            return False
        turn_key = self._turn_key(messages)
        if turn_key == _FORCED_TURN.get():
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

    def _turn_key(self, messages: list[BaseMessage]) -> str:
        index = self._latest_human_index(messages)
        human = messages[index]
        return str(getattr(human, "id", None) or f"{index}:{human.content!r}")

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

    def _retry_system_message(self, request: ModelRequest) -> SystemMessage:
        existing = request.system_message.text if request.system_message else ""
        content = f"{existing}\n\n{_RETRY_INSTRUCTIONS}".strip()
        return SystemMessage(content=content)


__all__ = [
    "DEFAULT_RETRIEVAL_RESULT_CHAR_LIMIT",
    "DEFAULT_RETRIEVAL_TURN_CHAR_LIMIT",
    "DocsResearchGuardMiddleware",
    "RESEARCH_TOOLS",
    "RETRIEVAL_TOOLS",
]
