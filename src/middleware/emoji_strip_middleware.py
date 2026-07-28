"""Egress guard: strip emoji from the docs agent's final answer.

The system prompt forbids emoji, but prompt-only enforcement drifts on long
and non-English answers. This middleware removes emoji code points from the
final assistant message while leaving fenced code blocks and inline code
spans (which may contain user-pasted content) untouched.
"""

from __future__ import annotations

import re
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.runtime import Runtime

#: Pictographic ranges covering decorative, status, and pointer glyphs.
EMOJI_PATTERN = re.compile(
    "[\U0001f300-\U0001faff\U0001f600-\U0001f64f"
    "\u2600-\u27bf\u2b00-\u2bff\ufe0f\u200d]"
)
#: Fenced code blocks and inline code spans are preserved verbatim.
CODE_PATTERN = re.compile(r"```.*?```|`[^`\n]*`", re.DOTALL)


def strip_emoji(text: str) -> str:
    """Remove emoji from prose, leaving code blocks and code spans unchanged."""
    out: list[str] = []
    cursor = 0
    for match in CODE_PATTERN.finditer(text):
        out.append(_strip_prose(text[cursor : match.start()]))
        out.append(match.group(0))
        cursor = match.end()
    out.append(_strip_prose(text[cursor:]))
    return "".join(out)


def _strip_prose(text: str) -> str:
    """Remove emoji and the whitespace runs their removal leaves behind."""
    stripped = EMOJI_PATTERN.sub("", text)
    if stripped == text:
        return text
    return re.sub(r"[ \t]{2,}", " ", stripped)


class EmojiStripMiddleware(AgentMiddleware):
    """Deterministically enforce the prompt's no-emoji response contract."""

    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Strip emoji from the final AI message before it reaches the user."""
        messages = state.get("messages", [])
        for message in reversed(messages):
            if getattr(message, "type", None) != "ai":
                continue
            cleaned = self._clean_content(message.content)
            if cleaned != message.content:
                # Same id => the messages reducer overwrites in place.
                message.content = cleaned
                return {"messages": [message]}
            break
        return None

    def _clean_content(self, content: Any) -> Any:
        """Strip emoji from string content or from text content blocks."""
        if isinstance(content, str):
            return strip_emoji(content)

        if not isinstance(content, list):
            return content

        cleaned: list[Any] = []
        for block in content:
            if isinstance(block, str):
                cleaned.append(strip_emoji(block))
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                cleaned.append({**block, "text": strip_emoji(block["text"])})
            else:
                cleaned.append(block)
        return cleaned


__all__ = ["EmojiStripMiddleware", "strip_emoji", "EMOJI_PATTERN"]
