"""Remove Mintlify authoring syntax from final documentation answers."""

from __future__ import annotations

import os
import re
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage

DOCS_MARKUP_SANITIZER_DISABLED_ENV = "DOCS_MARKUP_SANITIZER_DISABLED"

_FENCE_RE = re.compile(r"^(?P<indent>\s*)```(?P<info>[^\r\n]*)(?P<newline>\r?\n)?$")
_TAG_RE = re.compile(
    r"</?(?P<tag>Note|Info|Tip|Warning|Check|Accordion|AccordionGroup|CodeGroup|"
    r"Card|CardGroup|Steps|Step|Frame|Columns|Expandable)\b[^>]*>",
    re.IGNORECASE,
)
_LABELS = {
    "note": "Note",
    "info": "Info",
    "tip": "Tip",
    "warning": "Warning",
    "check": "Note",
}


class DocsMarkupSanitizerMiddleware(AgentMiddleware):
    """Sanitize Mintlify markup in the final AI response."""

    def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Sanitize the latest final AI message unless disabled."""
        if os.getenv(DOCS_MARKUP_SANITIZER_DISABLED_ENV, "").lower() in {
            "1",
            "true",
            "yes",
        }:
            return None
        messages = list(state.get("messages", []))
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if not isinstance(message, AIMessage) or message.tool_calls:
                continue
            content = message.content
            if not isinstance(content, str):
                continue
            sanitized = self._sanitize(content)
            if sanitized == content:
                return None
            return {
                "messages": [message.model_copy(update={"content": sanitized})]
            }
        return None

    def _sanitize(self, text: str) -> str:
        """Normalize fence headers and unwrap Mintlify components."""
        output: list[str] = []
        in_fence = False
        for line in text.splitlines(keepends=True):
            fence = _FENCE_RE.match(line)
            if fence:
                if in_fence:
                    output.append(line)
                    in_fence = False
                else:
                    info = fence.group("info").strip().split(maxsplit=1)
                    language = info[0] if info else ""
                    output.append(
                        f"{fence.group('indent')}```{language}{fence.group('newline') or ''}"
                    )
                    in_fence = True
                continue
            output.append(line if in_fence else _sanitize_components(line))
        return "".join(output)


def _sanitize_components(line: str) -> str:
    """Replace known Mintlify tags with plain markdown labels."""

    def replace(match: re.Match[str]) -> str:
        tag = match.group("tag").lower()
        if match.group(0).startswith("</"):
            return ""
        label = _LABELS.get(tag)
        return f"**{label}:** " if label else ""

    return _TAG_RE.sub(replace, line)
