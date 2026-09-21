"""Keep Python code examples consistent with the requested language."""

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

CODE_LANGUAGE_GUARD_DISABLED_ENV = "CODE_LANGUAGE_GUARD_DISABLED"
_DOCS_TOOLS = frozenset(
    {
        "search_docs_by_lang_chain",
        "query_docs_filesystem_docs_by_lang_chain",
    }
)
_FENCE_PATTERN = re.compile(r"```(?P<language>[A-Za-z0-9_+-]*)\s*\n(?P<body>.*?)(?:```|$)", re.DOTALL)
_JS_DECLARATION_PATTERN = re.compile(r"^\s*(?:const|let|var)\s+\w+\s*=", re.MULTILINE)
_JS_COMMENT_PATTERN = re.compile(r"^\s*//", re.MULTILINE)
_ARROW_PATTERN = re.compile(r"=>")
_CAMEL_IDENTIFIER_PATTERN = re.compile(r"\b[a-z][a-z0-9]*[A-Z][A-Za-z0-9]*\b")
_CAMEL_CASE_PATTERN = re.compile(
    r"\b([a-z][a-z0-9]*[A-Z][A-Za-z0-9]*)\s*[({]"
)
_PYTHON_PAGE_PATTERN = re.compile(
    r"Page:\s*`?(?:https?://[^\s`]+)?(?P<path>/(?:oss|langsmith)/python/[^\s`]+)",
    re.IGNORECASE,
)
_RETRY_INSTRUCTIONS = (
    "The previous response used JavaScript syntax in a Python example. Before answering, "
    "call search_docs_by_lang_chain with an explicit python language token, choose a Python "
    "documentation result, and read that exact returned Page path with "
    "query_docs_filesystem_docs_by_lang_chain. Rewrite the example as real Python. Do not "
    "derive a path or URL by replacing javascript with python or vice versa; use only paths "
    "returned verbatim by tools."
)
_DISCLOSURE = "A verified Python example could not be produced on this turn."
_MAX_FORCED_ATTEMPTS = 1
_FORCED_ATTEMPTS: contextvars.ContextVar[dict[str, int]] = contextvars.ContextVar(
    "code_language_guard_forced_attempts", default={}
)


class CodeLanguageGuardMiddleware(AgentMiddleware):
    """Reject JavaScript syntax in Python-fenced examples."""

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Retry once and sanitize invalid Python examples."""
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
            if not self._has_violation(
                self._response_messages(response),
                self._turn_messages(request.messages, self._response_messages(response)),
            ):
                self._clear_attempts(turn_key)
                return response

        self._clear_attempts(turn_key)
        return self._sanitize_response(request, response)

    def _should_retry(self, request: ModelRequest, response: ModelResponse) -> bool:
        if os.getenv(CODE_LANGUAGE_GUARD_DISABLED_ENV, "").lower() in {
            "1",
            "true",
            "yes",
        }:
            return False
        response_messages = self._response_messages(response)
        if self._has_pending_tool_calls(response_messages):
            return False
        return self._has_violation(
            response_messages,
            self._turn_messages(request.messages, response_messages),
        )

    def _has_violation(
        self, response_messages: list[BaseMessage], turn_messages: list[BaseMessage]
    ) -> bool:
        python_names = self._python_documentation_names(turn_messages)
        terminal_message = next(
            (
                message
                for message in reversed(response_messages)
                if isinstance(message, AIMessage)
            ),
            None,
        )
        if terminal_message is None:
            return False
        return any(
            self._block_is_invalid(block, python_names)
            for block in self._fenced_blocks(self._message_text(terminal_message))
        )

    def _block_is_invalid(self, block: tuple[str, str], python_names: set[str]) -> bool:
        language, body = block
        if language.lower() not in {"python", "py"}:
            return False
        if (
            _JS_DECLARATION_PATTERN.search(body)
            or _JS_COMMENT_PATTERN.search(body)
            or _ARROW_PATTERN.search(body)
        ):
            return True
        return any(
            match.group(0).split("(")[0].split("{")[0].strip() not in python_names
            for match in _CAMEL_CASE_PATTERN.finditer(body)
        )

    def _python_documentation_names(self, messages: list[BaseMessage]) -> set[str]:
        names: set[str] = set()
        for message in messages:
            if not isinstance(message, ToolMessage) or message.name not in _DOCS_TOOLS:
                continue
            text = self._message_text(message)
            if not _PYTHON_PAGE_PATTERN.search(text):
                continue
            names.update(_CAMEL_IDENTIFIER_PATTERN.findall(text))
        return names

    def _fenced_blocks(self, text: str) -> list[tuple[str, str]]:
        return [
            (match.group("language"), match.group("body"))
            for match in _FENCE_PATTERN.finditer(text)
        ]

    def _sanitize_response(
        self, request: ModelRequest, response: ModelResponse
    ) -> ModelResponse:
        messages = list(response.result)
        python_names = self._python_documentation_names(
            self._turn_messages(request.messages, [])
        )
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if not isinstance(message, AIMessage):
                continue
            text = self._message_text(message)
            sanitized = _FENCE_PATTERN.sub(
            lambda match: ""
            if self._block_is_invalid(
                (match.group("language"), match.group("body")), python_names
            )
            else match.group(0),
                text,
            )
            messages[index] = message.model_copy(
                update={"content": f"{_DISCLOSURE}\n\n{sanitized.strip()}"}
            )
            break
        return ModelResponse(
            result=messages,
            structured_response=response.structured_response,
        )

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
        return list(result) if result is not None else [response]

    def _has_pending_tool_calls(self, messages: list[BaseMessage]) -> bool:
        return any(
            isinstance(message, AIMessage) and bool(message.tool_calls)
            for message in messages
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
        return SystemMessage(content=f"{existing}\n\n{_RETRY_INSTRUCTIONS}".strip())

    def _attempt_count(self, turn_key: str) -> int:
        return _FORCED_ATTEMPTS.get().get(turn_key, 0)

    def _record_attempt(self, turn_key: str) -> None:
        attempts = dict(_FORCED_ATTEMPTS.get())
        attempts[turn_key] = attempts.get(turn_key, 0) + 1
        _FORCED_ATTEMPTS.set(attempts)

    def _clear_attempts(self, turn_key: str) -> None:
        attempts = dict(_FORCED_ATTEMPTS.get())
        attempts.pop(turn_key, None)
        _FORCED_ATTEMPTS.set(attempts)


__all__ = ["CodeLanguageGuardMiddleware"]
