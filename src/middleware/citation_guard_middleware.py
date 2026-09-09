"""Ensure documentation footers cite retrieved and valid URLs."""

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
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage

from src.tools.link_check_tools import _check_urls_async

DOCS_TOOLS = frozenset(
    {
        "search_docs_by_lang_chain",
        "query_docs_filesystem_docs_by_lang_chain",
    }
)
_URL_PATTERN = re.compile(r"https?://[^\s)<>]+")
_FOOTER_PATTERN = re.compile(r"(?ims)^\s*(?:\*\*)?Relevant docs:\s*(?:\*\*)?.*$")
_RETRY_INSTRUCTIONS = (
    "Rewrite the Relevant docs footer using only URLs copied verbatim from this turn's "
    "documentation tool results. Call check_links on exactly the final citation list "
    "before answering. Never construct or recall a documentation URL."
)


class CitationGuardMiddleware(AgentMiddleware):
    """Keep documentation citations grounded in current-turn evidence."""

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Validate and repair documentation citations after model calls."""
        response = await handler(request)
        if self._has_pending_tool_calls(self._response_messages(response)):
            return response

        latest_human_index = self._latest_human_index(request.messages)
        if latest_human_index < 0:
            return response
        turn_messages = request.messages[latest_human_index + 1 :]
        footer_message = self._footer_message(self._response_messages(response))
        if footer_message is None:
            return response

        footer_urls = self._urls_in_footer(footer_message)
        if not footer_urls:
            return response
        grounded_urls = self._grounded_urls(turn_messages)
        valid_urls = self._valid_urls(turn_messages)
        unchecked_urls = [
            url for url in footer_urls if url in grounded_urls and url not in valid_urls
        ]
        if unchecked_urls:
            results = await _check_urls_async(unchecked_urls, 10.0)
            valid_urls.update(result.url for result in results if result.valid)

        invalid_urls = {
            url
            for url in footer_urls
            if url not in grounded_urls or url not in valid_urls
        }
        if not invalid_urls:
            return response

        repaired_text = self._remove_footer_urls(
            self._message_text(footer_message), invalid_urls
        )
        if not self._urls_in_footer_text(repaired_text):
            retry_request = request.override(
                messages=[*request.messages, *self._response_messages(response)],
                system_message=self._retry_system_message(request, valid_urls),
            )
            return await handler(retry_request)
        return self._replace_footer(response, footer_message, repaired_text)

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

    def _footer_message(self, messages: list[BaseMessage]) -> AIMessage | None:
        for message in reversed(messages):
            if isinstance(
                message, AIMessage
            ) and "Relevant docs:" in self._message_text(message):
                return message
        return None

    def _urls_in_footer(self, message: AIMessage) -> list[str]:
        return self._urls_in_footer_text(self._message_text(message))

    def _urls_in_footer_text(self, text: str) -> list[str]:
        match = _FOOTER_PATTERN.search(text)
        return _URL_PATTERN.findall(match.group(0)) if match else []

    def _grounded_urls(self, messages: list[BaseMessage]) -> set[str]:
        return {
            url
            for message in messages
            if isinstance(message, ToolMessage) and message.name in DOCS_TOOLS
            for url in _URL_PATTERN.findall(self._message_text(message))
        }

    def _valid_urls(self, messages: list[BaseMessage]) -> set[str]:
        valid_urls: set[str] = set()
        for message in messages:
            if not isinstance(message, ToolMessage) or message.name != "check_links":
                continue
            in_valid_section = False
            for line in self._message_text(message).splitlines():
                stripped = line.strip()
                if stripped.lower() == "valid links:":
                    in_valid_section = True
                    continue
                if in_valid_section and stripped and not stripped.startswith("-"):
                    in_valid_section = False
                if in_valid_section:
                    valid_urls.update(_URL_PATTERN.findall(line))
        return valid_urls

    def _remove_footer_urls(self, text: str, invalid_urls: set[str]) -> str:
        match = _FOOTER_PATTERN.search(text)
        if not match:
            return text
        footer = match.group(0)
        lines = [
            line
            for line in footer.splitlines()
            if not invalid_urls.intersection(_URL_PATTERN.findall(line))
        ]
        return text[: match.start()] + "\n".join(lines) + text[match.end() :]

    def _replace_footer(
        self, response: ModelResponse, message: AIMessage, text: str
    ) -> ModelResponse:
        messages = self._response_messages(response)
        index = messages.index(message)
        messages[index] = message.model_copy(update={"content": text})
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

    def _retry_system_message(
        self, request: ModelRequest, valid_urls: set[str]
    ) -> SystemMessage:
        existing = request.system_message.text if request.system_message else ""
        validated = "\n".join(f"- {url}" for url in sorted(valid_urls)) or "- None"
        instructions = (
            f"{_RETRY_INSTRUCTIONS}\nURLs already validated on this turn:\n{validated}\n"
            "Do not request these URLs again."
        )
        return SystemMessage(content=f"{existing}\n\n{instructions}".strip())


__all__ = ["CitationGuardMiddleware"]
