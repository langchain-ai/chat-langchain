"""Drop citation URLs that were never retrieved or validated in the conversation."""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)

CHECK_LINKS_TOOL_NAME = "check_links"

#: Heading that opens the citation footer in the docs agent's answer format.
RELEVANT_DOCS_MARKER = "Relevant docs"

_URL_PATTERN = re.compile(r"https?://[^\s<>\"'\]\)}|]+")
_MARKDOWN_LINK_ENTRY = re.compile(r"^\s*[-*+]\s*\[[^\]]*\]\(\s*(?P<url>[^)\s]+)")
_RELEVANT_DOCS_HEADING = re.compile(
    rf"^\s*[*_#\s]*{RELEVANT_DOCS_MARKER}\b.*$", re.IGNORECASE
)
_VALID_SECTION_HEADING = "Valid links:"
_INVALID_SECTION_HEADING = "Invalid links:"


class LinkGroundingMiddleware(AgentMiddleware[AgentState]):
    """Strip "Relevant docs:" URLs the agent did not retrieve or validate."""

    def __init__(self, strip_unverified_links: bool = True):
        """Initialize the guard, optionally in log-only mode."""
        super().__init__()
        self.strip_unverified_links = strip_unverified_links
        logger.info(
            "LinkGroundingMiddleware initialized with strip_unverified_links=%s",
            self.strip_unverified_links,
        )

    def after_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Remove ungrounded citation URLs from the final answer."""
        messages = state.get("messages", [])
        if not messages:
            return None

        message = messages[-1]
        if not self._is_final_answer(message):
            return None

        text = message.content
        if not isinstance(text, str) or RELEVANT_DOCS_MARKER.lower() not in text.lower():
            return None

        grounded = self._grounded_urls(messages)
        footer_start = text.lower().index(RELEVANT_DOCS_MARKER.lower())
        ungrounded = [
            url
            for url in _URL_PATTERN.findall(text[footer_start:])
            if not self._is_grounded(url, grounded)
        ]
        if not ungrounded:
            return None

        logger.warning(
            "Ungrounded citation URLs in final response: %s (stripping=%s)",
            ", ".join(ungrounded),
            self.strip_unverified_links,
        )
        if not self.strip_unverified_links:
            return None

        cleaned = self._strip_ungrounded_entries(text, footer_start, grounded)
        if cleaned == text:
            return None

        # Same id => the messages reducer overwrites in place.
        message.content = cleaned
        return {"messages": [message]}

    def _is_final_answer(self, message: Any) -> bool:
        """Return True for an AI message that ends the turn."""
        return isinstance(message, AIMessage) and not getattr(
            message, "tool_calls", None
        )

    def _grounded_urls(self, messages: list[Any]) -> set[str]:
        """Collect URLs returned by tools plus URLs `check_links` reported valid."""
        check_links_args: dict[str, list[str]] = {}
        for message in messages:
            if not isinstance(message, AIMessage):
                continue
            for tool_call in getattr(message, "tool_calls", None) or []:
                if tool_call.get("name") != CHECK_LINKS_TOOL_NAME:
                    continue
                urls = (tool_call.get("args") or {}).get("urls") or []
                check_links_args[tool_call.get("id") or ""] = [
                    url for url in urls if isinstance(url, str)
                ]

        grounded: set[str] = set()
        for message in messages:
            if not isinstance(message, ToolMessage):
                continue
            content = message.content if isinstance(message.content, str) else str(
                message.content
            )
            if message.name == CHECK_LINKS_TOOL_NAME:
                requested = check_links_args.get(message.tool_call_id or "", [])
                for url in self._validated_urls(content, requested):
                    grounded.add(self._normalize(url))
                continue
            for url in _URL_PATTERN.findall(content):
                grounded.add(self._normalize(url))
        return grounded

    def _validated_urls(self, content: str, requested: list[str]) -> list[str]:
        """Return the requested URLs that a `check_links` result reported valid."""
        valid_section = content.split(_VALID_SECTION_HEADING)
        if len(valid_section) < 2:
            return []
        reported = valid_section[-1].split(_INVALID_SECTION_HEADING)[0]
        reported_urls = {
            self._normalize(url) for url in _URL_PATTERN.findall(reported)
        }
        candidates = requested or list(reported_urls)
        return [url for url in candidates if self._normalize(url) in reported_urls]

    def _normalize(self, url: str) -> str:
        """Normalize a URL for comparison by dropping trailing punctuation."""
        return url.rstrip(".,;:!?)\"'*").rstrip("/")

    def _is_grounded(self, url: str, grounded: set[str]) -> bool:
        """Check a URL against the grounded set, ignoring anchor fragments."""
        normalized = self._normalize(url)
        if normalized in grounded:
            return True
        base = self._normalize(normalized.split("#", 1)[0])
        return base in grounded

    def _strip_ungrounded_entries(
        self, text: str, footer_start: int, grounded: set[str]
    ) -> str:
        """Delete footer link entries whose URL is not grounded."""
        head, footer = text[:footer_start], text[footer_start:]
        kept: list[str] = []
        for line in footer.splitlines():
            match = _MARKDOWN_LINK_ENTRY.match(line)
            if match and not self._is_grounded(match.group("url"), grounded):
                continue
            kept.append(line)

        if not any(_MARKDOWN_LINK_ENTRY.match(line) for line in kept):
            kept = [
                line for line in kept if not _RELEVANT_DOCS_HEADING.match(line)
            ]
        return (head + "\n".join(kept)).rstrip() + "\n" if kept else head.rstrip() + "\n"


__all__ = ["LinkGroundingMiddleware", "RELEVANT_DOCS_MARKER"]
