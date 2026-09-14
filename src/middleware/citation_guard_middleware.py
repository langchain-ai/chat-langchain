"""Ensure documentation footers cite retrieved and valid URLs."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import unquote, urlsplit, urlunsplit

from deepagents.backends import FilesystemBackend
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
_OFFLOADED_PATH_PATTERN = re.compile(
    r"(?is)tool result too large.*?saved (?:in the filesystem )?at\s+"
    r"[\"'`]?([^\s\"'`<>]+)"
)
_FILESYSTEM_BACKEND = FilesystemBackend(root_dir="/", virtual_mode=False)
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
        for retry_count in range(3):
            response_messages = self._response_messages(response)
            footer_message = self._footer_message(response_messages)
            if footer_message is None:
                return response

            footer_urls = self._urls_in_footer(footer_message)
            if not footer_urls:
                return response
            evidence = [*turn_messages, *response_messages]
            grounded_urls = self._grounded_urls(evidence, footer_urls)
            valid_urls = self._valid_urls(evidence)
            invalid_reported_urls = self._invalid_urls(evidence)
            unchecked_urls = [
                url
                for url in footer_urls
                if url in grounded_urls
                and url not in valid_urls
                and url not in invalid_reported_urls
            ]
            if unchecked_urls:
                results = await _check_urls_async(unchecked_urls, 10.0)
                valid_urls.update(result.url for result in results if result.valid)

            invalid_urls = {
                url
                for url in footer_urls
                if url in invalid_reported_urls
                or url not in grounded_urls
                or url not in valid_urls
            }
            if not invalid_urls:
                return response

            repaired_text = self._remove_footer_urls(
                self._message_text(footer_message), invalid_urls
            )
            if self._urls_in_footer_text(repaired_text):
                return self._replace_footer(response, footer_message, repaired_text)
            if retry_count == 2:
                return self._replace_footer(
                    response,
                    footer_message,
                    self._remove_footer(self._message_text(footer_message)),
                )
            retry_request = request.override(
                messages=[*request.messages, *response_messages],
                system_message=self._retry_system_message(request),
            )
            response = await handler(retry_request)
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

    def _grounded_urls(
        self, messages: list[BaseMessage], candidate_urls: list[str] | None = None
    ) -> set[str]:
        evidence: dict[str, list[str]] = {}
        for message in messages:
            if not isinstance(message, ToolMessage) or message.name not in DOCS_TOOLS:
                continue
            text = self._docs_message_text(message)
            if text is None:
                continue
            urls = _URL_PATTERN.findall(text)
            body = _URL_PATTERN.sub("", text)
            for url in urls:
                evidence.setdefault(self._url_base(url), []).append(body)

        urls_to_check = candidate_urls or list(evidence)
        grounded: set[str] = set()
        for url in urls_to_check:
            parts = urlsplit(url)
            base = urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))
            bodies = evidence.get(base, [])
            if bodies and (
                not parts.fragment
                or any(
                    parts.fragment in body or unquote(parts.fragment) in body
                    for body in bodies
                )
            ):
                grounded.add(url)
        return grounded

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

    def _invalid_urls(self, messages: list[BaseMessage]) -> set[str]:
        invalid_urls: set[str] = set()
        for message in messages:
            if not isinstance(message, ToolMessage) or message.name != "check_links":
                continue
            in_invalid_section = False
            for line in self._message_text(message).splitlines():
                stripped = line.strip()
                if stripped.lower() == "invalid links:":
                    in_invalid_section = True
                    continue
                if in_invalid_section and stripped and not stripped.startswith("-"):
                    in_invalid_section = False
                if in_invalid_section:
                    invalid_urls.update(_URL_PATTERN.findall(line))
        return invalid_urls

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

    def _remove_footer(self, text: str) -> str:
        match = _FOOTER_PATTERN.search(text)
        return text[: match.start()] + text[match.end() :] if match else text

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

    def _docs_message_text(self, message: ToolMessage) -> str | None:
        text = self._message_text(message)
        match = _OFFLOADED_PATH_PATTERN.search(text)
        if not match:
            return text
        path = match.group(1).rstrip(".,;:")
        try:
            result = _FILESYSTEM_BACKEND.read(path, limit=100000)
        except (OSError, RuntimeError):
            return None
        if isinstance(result, str):
            return result
        file_data = getattr(result, "file_data", None)
        content = getattr(file_data, "content", None)
        encoding = getattr(file_data, "encoding", None)
        if isinstance(file_data, dict):
            content = file_data.get("content")
            encoding = file_data.get("encoding")
        if content is None and isinstance(result, dict):
            file_data = result.get("file_data", result)
            content = file_data.get("content") if isinstance(file_data, dict) else None
            encoding = (
                file_data.get("encoding") if isinstance(file_data, dict) else None
            )
        return (
            content
            if encoding in (None, "utf-8") and isinstance(content, str)
            else None
        )

    def _url_base(self, url: str) -> str:
        parts = urlsplit(url)
        return urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))

    def _content_part_text(self, part: Any) -> str:
        if isinstance(part, dict):
            return "\n".join(self._content_part_text(value) for value in part.values())
        if isinstance(part, list):
            return "\n".join(self._content_part_text(value) for value in part)
        return str(part)

    def _retry_system_message(self, request: ModelRequest) -> SystemMessage:
        existing = request.system_message.text if request.system_message else ""
        return SystemMessage(content=f"{existing}\n\n{_RETRY_INSTRUCTIONS}".strip())


__all__ = ["CitationGuardMiddleware"]
