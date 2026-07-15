"""Validate and repair language-tagged code fences in the final response.

docs_agent occasionally interpolates between multi-language documentation
examples and emits ``python``-tagged code blocks that contain JavaScript
arrow functions (``=>``) or C/C++ line comments (``//``) — both of which are
syntax errors in Python. This middleware runs after the agent finishes and
rewrites the offending syntax inside ``python`` fences (``//`` -> ``#``,
``(args) => expr`` -> ``lambda args: expr``) so copied snippets stay runnable.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)

#: Fence languages that must contain valid Python.
_PYTHON_FENCE_LANGS = {"python", "py", "python3"}

#: Matches a fenced code block, capturing the language tag and the body.
_FENCE_RE = re.compile(
    r"(?P<fence>```|~~~)[ \t]*(?P<lang>[A-Za-z0-9_+-]*)[ \t]*\n"
    r"(?P<body>.*?)(?P=fence)",
    re.DOTALL,
)

#: Python string literals (single/double, incl. triple-quoted) for masking.
_PY_STRING_RE = re.compile(
    r"'''.*?'''|\"\"\".*?\"\"\"|'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"",
    re.DOTALL,
)

#: A ``//`` line comment that is not part of ``://`` (URLs) or ``//=`` (floordiv).
_JS_COMMENT_RE = re.compile(r"(?<![:/])//(?!=)[^\n]*")

#: Simple single-expression arrow function: ``(a, b) => expr`` or ``x => expr``.
_ARROW_RE = re.compile(
    r"\(([^()]*)\)\s*=>\s*([^\n;{}]+)|([A-Za-z_]\w*)\s*=>\s*([^\n;{}]+)"
)


class CodeFenceValidatorMiddleware(AgentMiddleware):
    """Repair non-Python syntax leaking into ``python`` code fences."""

    def after_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Sanitize python fences in the final AI message before it is emitted."""
        messages = state.get("messages", [])
        if not messages:
            return None

        last = messages[-1]
        if getattr(last, "type", None) != "ai":
            return None

        repaired = self._repair_content(last.content)
        if repaired is last.content:
            return None

        # Reuse the same id so the messages reducer overwrites in place.
        return {"messages": [AIMessage(id=last.id, content=repaired)]}

    def _repair_content(self, content: Any) -> Any:
        """Repair python fences in either a string or a list of content blocks."""
        if isinstance(content, str):
            return self._repair_text(content)

        if not isinstance(content, list):
            return content

        changed = False
        repaired_blocks: list[Any] = []
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ):
                new_text = self._repair_text(block["text"])
                changed = changed or new_text != block["text"]
                repaired_blocks.append({**block, "text": new_text})
            else:
                repaired_blocks.append(block)
        return repaired_blocks if changed else content

    def _repair_text(self, text: str) -> str:
        """Rewrite non-Python syntax inside every python fence in ``text``."""

        def _replace(match: re.Match[str]) -> str:
            lang = match.group("lang").lower()
            if lang not in _PYTHON_FENCE_LANGS:
                return match.group(0)
            body = match.group("body")
            fixed = self._repair_python_body(body)
            if fixed == body:
                return match.group(0)
            logger.warning("Rewrote non-Python syntax inside a python code fence")
            return f"{match.group('fence')}{match.group('lang')}\n{fixed}{match.group('fence')}"

        return _FENCE_RE.sub(_replace, text)

    def _repair_python_body(self, body: str) -> str:
        """Convert ``//`` comments and simple ``=>`` arrows to valid Python."""
        # Mask string literals so we don't rewrite ``//`` or ``=>`` inside them.
        masks: list[str] = []

        def _mask(match: re.Match[str]) -> str:
            masks.append(match.group(0))
            return f"\x00{len(masks) - 1}\x00"

        masked = _PY_STRING_RE.sub(_mask, body)

        masked = _JS_COMMENT_RE.sub(lambda m: "#" + m.group(0)[2:], masked)

        def _arrow(match: re.Match[str]) -> str:
            if match.group(1) is not None:
                params, expr = match.group(1), match.group(2)
            else:
                params, expr = match.group(3), match.group(4)
            return f"lambda {params.strip()}: {expr.strip()}"

        masked = _ARROW_RE.sub(_arrow, masked)

        return re.sub(r"\x00(\d+)\x00", lambda m: masks[int(m.group(1))], masked)


__all__ = ["CodeFenceValidatorMiddleware"]
