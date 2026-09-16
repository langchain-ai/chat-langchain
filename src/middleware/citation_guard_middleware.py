"""Ensure documentation footers cite retrieved and valid URLs."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentState
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ExtendedModelResponse,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.types import Command
from typing_extensions import NotRequired

from src.tools.link_check_tools import _check_urls_async

DOCS_TOOLS = frozenset(
    {
        "search_docs_by_lang_chain",
        "query_docs_filesystem_docs_by_lang_chain",
    }
)
_URL_PATTERN = re.compile(r"https?://[^\s)<>]+")
_FOOTER_PATTERN = re.compile(
    r"(?ims)^\s*(?:(?:\*\*)?Relevant docs:\s*(?:\*\*)?|##\s+Relevant docs:\s*).*$"
)
_RETRY_INSTRUCTIONS = (
    "Rewrite the Relevant docs footer using only URLs copied verbatim from this turn's "
    "documentation tool results. Call check_links on exactly the final citation list "
    "before answering. Never construct or recall a documentation URL."
)
_MAX_REPAIR_ATTEMPTS = 2


class CitationGuardState(AgentState):
    """State fields used by the citation guard."""

    citation_guard_attempts: NotRequired[dict[str, int]]


class CitationGuardMiddleware(AgentMiddleware):
    """Keep documentation citations grounded in current-turn evidence."""

    state_schema = CitationGuardState

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
        turn_key = self._turn_key(request.messages)
        attempts = self._attempts(request.state)
        turn_messages = request.messages[latest_human_index + 1 :]
        footer_message = self._footer_message(self._response_messages(response))
        if footer_message is None:
            attempts.pop(turn_key, None)
            return self._stateful_response(request, response, attempts)

        footer_urls = self._urls_in_footer(footer_message)
        if not footer_urls:
            attempts.pop(turn_key, None)
            return self._stateful_response(request, response, attempts)
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
            attempts.pop(turn_key, None)
            return self._stateful_response(request, response, attempts)

        repaired_text = self._remove_footer_urls(
            self._message_text(footer_message), invalid_urls
        )
        if self._urls_in_footer_text(repaired_text):
            attempts.pop(turn_key, None)
            return self._stateful_response(
                request,
                self._replace_footer(response, footer_message, repaired_text),
                attempts,
            )
        if attempts.get(turn_key, 0) >= _MAX_REPAIR_ATTEMPTS:
            attempts.pop(turn_key, None)
            return self._stateful_response(
                request,
                self._remove_footer(response, footer_message),
                attempts,
            )
        attempts[turn_key] = attempts.get(turn_key, 0) + 1
        retry_request = request.override(
            messages=[
                *request.messages,
                HumanMessage(
                    content=_RETRY_INSTRUCTIONS,
                    additional_kwargs={"guard_injected": True},
                ),
            ],
            system_message=self._retry_system_message(request),
        )
        retry_response = await handler(retry_request)
        return self._stateful_response(request, retry_response, attempts)

    def _latest_human_index(self, messages: list[BaseMessage]) -> int:
        for index in range(len(messages) - 1, -1, -1):
            if self._is_real_human(messages[index]):
                return index
        return -1

    def _is_real_human(self, message: BaseMessage) -> bool:
        return getattr(message, "type", None) == "human" and not getattr(
            message, "additional_kwargs", {}
        ).get("guard_injected", False)

    def _turn_key(self, messages: list[BaseMessage]) -> str:
        index = self._latest_human_index(messages)
        if index < 0:
            return "no-human-message"
        human = messages[index]
        return f"{index}:{getattr(human, 'id', None) or human.content!r}"

    def _attempts(self, state: Any) -> dict[str, int]:
        attempts = state.get("citation_guard_attempts", {})
        return dict(attempts)

    def _stateful_response(
        self, request: ModelRequest, response: ModelResponse, attempts: dict[str, int]
    ) -> ModelCallResult:
        attempts = {key: value for key, value in attempts.items() if value is not None}
        update = {"citation_guard_attempts": attempts}
        if request.runtime is None:
            request.state.update(update)
            return response
        return ExtendedModelResponse(response, Command(update=update))

    def _response_messages(self, response: ModelResponse) -> list[BaseMessage]:
        result = getattr(response, "result", None)
        return list(result) if result is not None else []

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
        content = message.content
        if isinstance(content, list):
            updated_content = list(content)
            for part_index, part in enumerate(updated_content):
                if not isinstance(
                    part, dict
                ) or "Relevant docs:" not in self._content_part_text(part):
                    continue
                updated_content[part_index] = {**part, "text": text}
                break
            content = updated_content
        else:
            content = text
        messages[index] = message.model_copy(update={"content": content})
        response.result = messages
        return response

    def _remove_footer(
        self, response: ModelResponse, message: AIMessage
    ) -> ModelResponse:
        return self._replace_footer(response, message, "")

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

    def _retry_system_message(self, request: ModelRequest) -> SystemMessage:
        existing = request.system_message.text if request.system_message else ""
        return SystemMessage(content=f"{existing}\n\n{_RETRY_INSTRUCTIONS}".strip())


__all__ = ["CitationGuardMiddleware"]
