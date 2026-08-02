"""Output guard: repair or drop answer links on unapproved hostnames.

``check_links`` rejects unapproved hosts, but the model can skip validation or
retype a hostname after validating it. This middleware is the deterministic
backstop: no citation on a host LangChain does not own reaches the user.
"""

from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import urlparse, urlunparse

from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.runtime import Runtime

from src.tools.link_check_tools import ALLOWED_DOC_HOSTS

logger = logging.getLogger(__name__)

#: Host used when an unapproved link still points at a recognizable docs path.
CANONICAL_DOC_HOST = "docs.langchain.com"

#: Path prefixes served by ``docs.langchain.com``.
DOC_PATH_PREFIXES = ("/oss/", "/langsmith/", "/langgraph/")

_MARKDOWN_LINK = re.compile(r"\[([^\]]*)\]\((https?://[^\s)]+)\)")
_BARE_URL = re.compile(r"https?://[^\s)\]]+")


class LinkHostGuardMiddleware(AgentMiddleware):
    """Rewrite or remove links whose host is not an approved documentation domain."""

    def after_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Sanitize link hosts in the final assistant message."""
        messages = state.get("messages", [])
        if not messages:
            return None

        message = messages[-1]
        if getattr(message, "type", None) != "ai":
            return None

        sanitized = self._sanitize_content(message.content)
        if sanitized is message.content:
            return None

        # Same id => the messages reducer overwrites in place.
        message.content = sanitized
        return {"messages": [message]}

    def _sanitize_content(self, content: Any) -> Any:
        """Sanitize plain-string content or a list of content blocks."""
        if isinstance(content, str):
            sanitized = self._sanitize_text(content)
            return sanitized if sanitized != content else content

        if not isinstance(content, list):
            return content

        changed = False
        blocks: list[Any] = []
        for block in content:
            if isinstance(block, str):
                sanitized = self._sanitize_text(block)
                changed = changed or sanitized != block
                blocks.append(sanitized)
            elif (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ):
                sanitized = self._sanitize_text(block["text"])
                changed = changed or sanitized != block["text"]
                blocks.append({**block, "text": sanitized})
            else:
                blocks.append(block)
        return blocks if changed else content

    def _sanitize_text(self, text: str) -> str:
        """Repair or drop every unapproved URL in a block of answer text."""
        def _replace_markdown(match: re.Match[str]) -> str:
            fixed = self._fix_url(match.group(2))
            return f"[{match.group(1)}]({fixed})" if fixed else match.group(1)

        text = _MARKDOWN_LINK.sub(_replace_markdown, text)
        return _BARE_URL.sub(lambda m: self._fix_url(m.group(0)) or "", text)

    def _fix_url(self, url: str) -> str | None:
        """Return an approved-host URL, or None when the link must be dropped."""
        try:
            parts = urlparse(url)
        except Exception:
            logger.warning("Dropping unparseable link in answer")
            return None

        host = parts.netloc.lower()
        if host in ALLOWED_DOC_HOSTS:
            return url

        if parts.path.startswith(DOC_PATH_PREFIXES):
            logger.warning(
                "Rewriting citation host %s to %s", host, CANONICAL_DOC_HOST
            )
            return urlunparse(parts._replace(netloc=CANONICAL_DOC_HOST))

        logger.warning("Dropping citation on unapproved host %s", host)
        return None


__all__ = [
    "CANONICAL_DOC_HOST",
    "DOC_PATH_PREFIXES",
    "LinkHostGuardMiddleware",
]
