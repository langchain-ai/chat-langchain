"""Validate executable code blocks in terminal answers."""

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
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

CODE_BLOCK_GUARD_DISABLED_ENV = "CODE_BLOCK_GUARD_DISABLED"
_CODE_BLOCK_PATTERN = re.compile(
    r"```(?P<language>[^\n`]*)\n(?P<code>.*?)```", re.DOTALL
)
_FORCED_TURN: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "code_block_guard_forced_turn", default=None
)


class CodeBlockGuardMiddleware(AgentMiddleware):
    """Validate and repair executable code blocks in terminal answers."""

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Validate executable blocks and retry one invalid answer."""
        response = await handler(request)
        response_messages = self._response_messages(response)
        if self._has_pending_tool_calls(response_messages):
            return response
        if os.getenv(CODE_BLOCK_GUARD_DISABLED_ENV, "").lower() in {"1", "true", "yes"}:
            return response

        terminal_message = self._terminal_message(response_messages)
        if terminal_message is None:
            return response
        failures = self._validate_blocks(self._message_text(terminal_message))
        if not failures:
            return response

        turn_key = self._turn_key(request.messages)
        if turn_key != _FORCED_TURN.get():
            _FORCED_TURN.set(turn_key)
            instruction = self._retry_instruction(failures)
            retry_request = request.override(
                messages=[*request.messages, HumanMessage(content=instruction)]
            )
            retry_response = await handler(retry_request)
            retry_message = self._terminal_message(
                self._response_messages(retry_response)
            )
            if retry_message is not None and not self._validate_blocks(
                self._message_text(retry_message)
            ):
                return retry_response
            return self._strip_invalid_blocks(retry_response)

        return self._strip_invalid_blocks(response)

    def _latest_human_index(self, messages: list[BaseMessage]) -> int:
        for index in range(len(messages) - 1, -1, -1):
            if getattr(messages[index], "type", None) == "human":
                return index
        return -1

    def _turn_key(self, messages: list[BaseMessage]) -> str:
        index = self._latest_human_index(messages)
        if index < 0:
            return ""
        human = messages[index]
        return str(getattr(human, "id", None) or f"{index}:{human.content!r}")

    def _response_messages(self, response: ModelResponse) -> list[BaseMessage]:
        result = getattr(response, "result", None)
        return list(result) if result is not None else [response]

    def _has_pending_tool_calls(self, messages: list[BaseMessage]) -> bool:
        return any(
            isinstance(message, AIMessage) and bool(message.tool_calls)
            for message in messages
        )

    def _terminal_message(self, messages: list[BaseMessage]) -> AIMessage | None:
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                return message
        return None

    def _message_text(self, message: BaseMessage) -> str:
        content: Any = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                text = block.get("text", block.get("content", ""))
                if isinstance(text, str):
                    parts.append(text)
            return "\n".join(parts)
        return str(content)

    def _validate_blocks(self, text: str) -> list[tuple[int, str, str]]:
        failures = []
        for index, match in enumerate(_CODE_BLOCK_PATTERN.finditer(text), start=1):
            language = match.group("language").strip().lower()
            code = match.group("code")
            error = self._validate_block(language, code)
            if error:
                failures.append(
                    (index, f"block {index} ({language or 'unspecified'})", error)
                )
        return failures

    def _validate_block(self, language: str, code: str) -> str | None:
        if language in {"python", "py"}:
            try:
                compile(code, "<answer>", "exec")
            except SyntaxError as exc:
                return str(exc)
        if language in {"bash", "sh", "shell"}:
            activation = re.search(
                r"(?:^|[;&|]\s*)source\s+([^\s;&|]+)", code, re.MULTILINE
            )
            if activation:
                path = activation.group(1)
                if not re.search(r"(?:^|/)bin/activate$", path) and not re.search(
                    r"(?:^|[\\/])Scripts\\activate$", path
                ):
                    return f"invalid virtualenv activation path: {path}"
            for command in re.finditer(
                r"(?:^|[;&|]\s*)(?:python\s+-m\s+)?pip\s+install|"
                r"(?:^|[;&|]\s*)uv\s+add\b",
                code,
                re.MULTILINE,
            ):
                command_text = code[command.end() :].splitlines()[0]
                arguments = command_text.split("#", 1)[0].strip()
                if not arguments or all(
                    argument.startswith("-") for argument in arguments.split()
                ):
                    command_name = (
                        "pip install" if "pip" in command.group() else "uv add"
                    )
                    return f"{command_name} requires a package target"
        return None

    def _retry_instruction(self, failures: list[tuple[int, str, str]]) -> str:
        details = "; ".join(f"{block}: {error}" for _, block, error in failures)
        return (
            "Rewrite the answer so every executable code block passes validation. "
            f"Fix these failures exactly: {details}"
        )

    def _strip_invalid_blocks(self, response: ModelResponse) -> ModelResponse:
        messages = self._response_messages(response)
        terminal_message = self._terminal_message(messages)
        if terminal_message is None:
            return response
        text = self._message_text(terminal_message)
        failures = self._validate_blocks(text)
        if not failures:
            return response
        failure_by_index = {index: error for index, _, error in failures}
        block_index = 0

        def replace(match: re.Match[str]) -> str:
            nonlocal block_index
            block_index += 1
            error = failure_by_index.get(block_index)
            if error is None:
                return match.group(0)
            return f"[Code block removed because validation failed: {error}]"

        sanitized = _CODE_BLOCK_PATTERN.sub(replace, text)
        index = messages.index(terminal_message)
        messages[index] = terminal_message.model_copy(update={"content": sanitized})
        response.result = messages
        return response


__all__ = ["CodeBlockGuardMiddleware"]
