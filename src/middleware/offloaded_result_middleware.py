"""Re-inline tool results the harness offloaded to ``/large_tool_results/``.

When a tool result exceeds the runtime size cap, the Deep Agents harness swaps
the payload for a pointer ("... was saved in the filesystem at this path:
/large_tool_results/<id>") and expects the model to fetch it with ``read_file``.
Models routinely ignore that instruction and answer from memory, so this
middleware resolves the pointer back into content before the next model call.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)

#: Marker the harness writes into an offloaded tool result.
OFFLOAD_MARKER = "/large_tool_results/"

OFFLOAD_PATH_PATTERN = re.compile(r"/large_tool_results/[^\s\"'`)\]]+")

#: Upper bound on re-inlined content, keeping a re-read escape hatch.
MAX_INLINE_CHARS = 60_000

CONTINUATION_NOTICE = (
    '\n\n[Truncated after {chars} characters. Call read_file(file_path="{path}") '
    "for the full result.]"
)


class OffloadedToolResultMiddleware(AgentMiddleware):
    """Replace offloaded tool-result pointers with the payload they point at."""

    def __init__(self, max_inline_chars: int = MAX_INLINE_CHARS) -> None:
        """Configure how much re-inlined content is kept per tool message."""
        super().__init__()
        self.max_inline_chars = max_inline_chars

    def before_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Resolve offload pointers in the most recent batch of tool messages."""
        files = state.get("files") or {}
        updated: list[Any] = []
        for message in reversed(state.get("messages", [])):
            if getattr(message, "type", None) != "tool":
                break
            resolved = self._resolve(message.content, files)
            if resolved is not None:
                # Same id => the messages reducer overwrites in place.
                message.content = resolved
                updated.append(message)
        return {"messages": updated} if updated else None

    def _resolve(self, content: Any, files: Any) -> str | None:
        """Return the payload for an offloaded tool result, else ``None``."""
        if not isinstance(content, str) or OFFLOAD_MARKER not in content:
            return None

        match = OFFLOAD_PATH_PATTERN.search(content)
        if match is None:
            return None

        path = match.group(0)
        payload = self._read(path, files)
        if not payload:
            logger.warning("Could not re-inline offloaded tool result at %s", path)
            return None

        if len(payload) <= self.max_inline_chars:
            return payload
        return payload[: self.max_inline_chars] + CONTINUATION_NOTICE.format(
            chars=self.max_inline_chars, path=path
        )

    def _read(self, path: str, files: Any) -> str | None:
        """Read the offloaded payload from the agent filesystem or local disk."""
        if isinstance(files, dict):
            for key in (path, path.lstrip("/")):
                payload = self._entry_text(files.get(key))
                if payload is not None:
                    return payload

        try:
            return Path(path).read_text()
        except OSError:
            return None

    def _entry_text(self, entry: Any) -> str | None:
        """Normalize a filesystem entry into text."""
        if isinstance(entry, str):
            return entry
        if isinstance(entry, dict):
            content = entry.get("content")
        else:
            content = getattr(entry, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(str(line) for line in content)
        return None


__all__ = [
    "MAX_INLINE_CHARS",
    "OFFLOAD_MARKER",
    "OffloadedToolResultMiddleware",
]
