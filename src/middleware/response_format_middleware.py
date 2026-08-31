"""Normalize comment syntax in generated Python code examples."""

from __future__ import annotations

import io
import logging
import re
import tokenize
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState

logger = logging.getLogger(__name__)

_FENCED_BLOCK = re.compile(
    r"(?P<opening>```(?P<language>[^\n`]*)\n)(?P<code>.*?)(?P<closing>```)",
    re.DOTALL,
)
_PYTHON_INDICATORS = re.compile(r"(?m)^\s*(?:var|const|let)\b|=>")
_LEADING_JS_COMMENT = re.compile(r"^(?P<indent>\s*)//(?=\s|$)")


def _string_lines(code: str) -> set[int]:
    lines: set[int] = set()
    try:
        tokens = tokenize.generate_tokens(io.StringIO(code).readline)
        for token in tokens:
            if token.type != tokenize.STRING:
                continue
            lines.update(range(token.start[0], token.end[0] + 1))
    except (IndentationError, tokenize.TokenError):
        return lines
    return lines


def normalize_python_fences(text: str, *, warning_logger: Any = logger) -> str:
    """Convert leading JavaScript comments in Python fences to Python syntax."""
    def replace_block(match: re.Match[str]) -> str:
        language = match.group("language").strip().lower()
        code = match.group("code")
        if language not in {"python", "py"}:
            return match.group(0)
        if _PYTHON_INDICATORS.search(code):
            warning_logger.warning(
                "Python code fence contains JavaScript indicators"
            )
        string_lines = _string_lines(code)
        normalized_lines = []
        for offset, line in enumerate(code.splitlines(keepends=True)):
            if offset + 1 not in string_lines:
                line = _LEADING_JS_COMMENT.sub(r"\g<indent>#", line)
            normalized_lines.append(line)
        return f"{match.group('opening')}{''.join(normalized_lines)}{match.group('closing')}"

    return _FENCED_BLOCK.sub(replace_block, text)


def _normalize_message(message: Any) -> Any:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        normalized = normalize_python_fences(content)
    elif isinstance(content, list):
        normalized_blocks = []
        for block in content:
            if isinstance(block, str):
                normalized_blocks.append(normalize_python_fences(block))
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                normalized_blocks.append(
                    {**block, "text": normalize_python_fences(block["text"])}
                )
            else:
                normalized_blocks.append(block)
        normalized = normalized_blocks
    else:
        return message
    if normalized == content:
        return message
    return message.model_copy(update={"content": normalized})


class ResponseFormatMiddleware(AgentMiddleware[AgentState]):
    """Normalize the final model message before it is emitted."""

    def after_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        """Normalize fenced code in the latest model message."""
        messages = state.get("messages", [])
        if not messages:
            return None
        message = messages[-1]
        normalized = _normalize_message(message)
        return None if normalized is message else {"messages": [normalized]}

    async def aafter_model(
        self, state: AgentState, runtime: Any
    ) -> dict[str, Any] | None:
        """Normalize fenced code in the latest async model message."""
        return self.after_model(state, runtime)


__all__ = ["ResponseFormatMiddleware", "normalize_python_fences"]
