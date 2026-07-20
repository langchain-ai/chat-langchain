"""Egress guard: strip fabricated non-langchain.com documentation links.

The model (``gemini-3.1-flash-lite``) sometimes emits documentation URLs from
its parametric knowledge that point at third-party domains (e.g.
``agentskills.io``) even though those URLs never appeared in any tool result.
This middleware enforces a domain allowlist over the final answer as a
defense-in-depth backstop to the instructions in ``instructions.md``: any
markdown link whose host is not ``langchain.com`` or a subdomain of it is
demoted to plain text so the fabricated URL is never presented as a citation.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

#: Root domain (and its subdomains) allowed in documentation links.
ALLOWED_ROOT_DOMAIN = "langchain.com"

_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\((\s*<?)(https?://[^\s)>]+)(>?\s*)\)")


def _host_allowed(url: str) -> bool:
    """Return True when the URL host is langchain.com or a subdomain of it."""
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    if not host:
        return False
    return host == ALLOWED_ROOT_DOMAIN or host.endswith("." + ALLOWED_ROOT_DOMAIN)


def _strip_disallowed_links(text: str) -> str:
    """Replace markdown links to non-allowlisted hosts with their link text."""

    def _replace(match: re.Match[str]) -> str:
        label, url = match.group(1), match.group(3)
        return match.group(0) if _host_allowed(url) else label

    return _MARKDOWN_LINK.sub(_replace, text)


class LinkAllowlistMiddleware(AgentMiddleware):
    """Demote markdown links to non-langchain.com hosts in the final answer."""

    def after_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Strip fabricated non-allowlisted documentation links from the answer."""
        messages = state.get("messages", [])
        for message in reversed(messages):
            if not isinstance(message, AIMessage):
                continue
            filtered = self._filter_content(message.content)
            if filtered is not message.content:
                message.content = filtered
                return {"messages": [message]}
            return None
        return None

    def _filter_content(self, content: Any) -> Any:
        """Strip disallowed links from a str or list-of-blocks message content."""
        if isinstance(content, str):
            filtered = _strip_disallowed_links(content)
            return filtered if filtered != content else content

        if not isinstance(content, list):
            return content

        changed = False
        blocks: list[Any] = []
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ):
                filtered = _strip_disallowed_links(block["text"])
                changed = changed or filtered != block["text"]
                blocks.append({**block, "text": filtered})
            else:
                blocks.append(block)
        return blocks if changed else content


__all__ = ["LinkAllowlistMiddleware", "ALLOWED_ROOT_DOMAIN"]
