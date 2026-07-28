"""Citation provenance: enforce grounding of the ``Relevant docs:`` footer.

The system prompt asks the model to validate links with ``check_links`` before
sending, but that guard is advisory and is applied inconsistently — traces show
footer URLs that no retrieval tool ever returned and that were never passed to
``check_links``. This middleware turns the guard into a postcondition: any
citation in the trailing ``Relevant docs:`` block with no provenance anywhere in
the run is dropped from the final answer.

Nothing here touches the network. ``check_links`` remains the model's tool; this
middleware only reads what the run already produced.
"""

from __future__ import annotations

import re
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.runtime import Runtime

#: Hosts whose URLs the agent is expected to have retrieved rather than recalled.
CITED_HOSTS = (
    "docs.langchain.com",
    "support.langchain.com",
    "reference.langchain.com",
)

_FOOTER_RE = re.compile(r"^[ \t]*\*{0,2}relevant docs:?\*{0,2}[ \t]*:?[ \t]*$")
_LIST_ITEM_RE = re.compile(r"^[ \t]*[-*+][ \t]+")
_MARKDOWN_LINK_RE = re.compile(r"\[[^\]]*\]\(\s*(<?)([^)\s]+)\1\s*\)")
_URL_RE = re.compile(r"https?://[^\s)\]\"'`<>,]+")


class CitationProvenanceMiddleware(AgentMiddleware):
    """Drop ``Relevant docs:`` citations with no provenance in this run."""

    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Prune ungrounded citations from the final answer's docs footer."""
        messages = state.get("messages", [])
        final = self._final_ai_message(messages)
        if final is None:
            return None

        corroboration = self._corroboration_text(messages, final)
        pruned = self._prune_footer(self._text(final), corroboration)
        if pruned is None:
            return None

        # Same id => the messages reducer overwrites in place.
        final.content = self._replace_text(final.content, pruned)
        return {"messages": [final]}

    def _replace_text(self, content: Any, pruned: str) -> Any:
        """Put the pruned text back into the message's original content shape."""
        if not isinstance(content, list):
            return pruned

        rebuilt: list[Any] = []
        written = False
        for block in content:
            is_str = isinstance(block, str)
            is_text_block = (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            )
            if not is_str and not is_text_block:
                rebuilt.append(block)
                continue
            if written:
                continue
            written = True
            rebuilt.append(pruned if is_str else {**block, "text": pruned})
        return rebuilt if written else pruned

    def _final_ai_message(self, messages: list[Any]) -> Any | None:
        """Return the last AI message carrying non-empty text content."""
        for message in reversed(messages):
            if getattr(message, "type", None) != "ai":
                continue
            if self._text(message).strip():
                return message
        return None

    def _text(self, message: Any) -> str:
        """Flatten a message's content into plain text."""
        return self._content_text(getattr(message, "content", None))

    def _content_text(self, content: Any) -> str:
        """Flatten ``str`` or content-block-list content into plain text."""
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""

        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
                elif isinstance(block.get("content"), (str, list)):
                    parts.append(self._content_text(block["content"]))
        return "\n".join(parts)

    def _corroboration_text(self, messages: list[Any], final: Any) -> str:
        """Collect every piece of run text a citation may be grounded in."""
        parts: list[str] = []
        for message in messages:
            if message is final:
                continue

            kind = getattr(message, "type", None)
            if kind in ("tool", "human"):
                parts.append(self._content_text(getattr(message, "content", None)))
            elif kind == "ai":
                for call in getattr(message, "tool_calls", None) or []:
                    args = call.get("args") if isinstance(call, dict) else None
                    if isinstance(args, dict):
                        parts.append(self._flatten_args(args))
        return "\n".join(parts).lower()

    def _flatten_args(self, args: dict[str, Any]) -> str:
        """Flatten tool-call args (notably ``check_links`` urls) into text."""
        parts: list[str] = []
        for value in args.values():
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, (list, tuple)):
                parts.extend(str(item) for item in value)
            elif isinstance(value, dict):
                parts.append(self._flatten_args(value))
        return "\n".join(parts)

    def _prune_footer(self, text: str, corroboration: str) -> str | None:
        """Rewrite the answer without ungrounded footer citations, or ``None``."""
        lines = text.split("\n")
        header = self._footer_header_index(lines)
        if header is None:
            return None

        kept: list[str] = []
        removed = 0
        citations = 0
        for index, line in enumerate(lines):
            if index <= header or not _LIST_ITEM_RE.match(line):
                kept.append(line)
                continue
            urls = self._cited_urls(line)
            if not urls:
                kept.append(line)
                continue
            citations += 1
            if all(self._is_grounded(url, corroboration) for url in urls):
                kept.append(line)
            else:
                removed += 1

        if not removed:
            return None

        if removed == citations:
            kept = self._drop_empty_footer(kept, header)
        return "\n".join(kept)

    def _footer_header_index(self, lines: list[str]) -> int | None:
        """Return the index of the trailing ``Relevant docs:`` header line."""
        for index in range(len(lines) - 1, -1, -1):
            if _FOOTER_RE.match(lines[index].lower()):
                return index
        return None

    def _cited_urls(self, line: str) -> list[str]:
        """Extract the checkable URLs a footer list item points at."""
        urls = [match.group(2) for match in _MARKDOWN_LINK_RE.finditer(line)]
        urls.extend(_URL_RE.findall(line))
        return [url for url in urls if self._is_cited_host(url)]

    def _is_cited_host(self, url: str) -> bool:
        """Report whether the URL belongs to a docs host we can corroborate."""
        lowered = url.lower()
        return any(f"//{host}" in lowered for host in CITED_HOSTS)

    def _is_grounded(self, url: str, corroboration: str) -> bool:
        """Report whether the URL (or its docs path) appeared earlier in the run."""
        for candidate in self._url_variants(url):
            if candidate and candidate in corroboration:
                return True
        return False

    def _url_variants(self, url: str) -> list[str]:
        """Build the comparable forms of a cited URL, anchor-stripped included."""
        lowered = url.lower().split("#", 1)[0].rstrip("/")
        variants = [lowered]
        for host in CITED_HOSTS:
            marker = f"//{host}"
            if marker in lowered:
                # Docs filesystem paths (e.g. `/oss/python/foo.mdx`) corroborate
                # the public URL the model derived from them.
                path = lowered.split(marker, 1)[1]
                if path:
                    variants.append(path)
                break
        return variants

    def _drop_empty_footer(self, lines: list[str], header: int) -> list[str]:
        """Remove a ``Relevant docs:`` header left with no surviving citations."""
        end = len(lines)
        while end > header + 1 and not lines[end - 1].strip():
            end -= 1
        if end != header + 1:
            return lines
        start = header
        while start > 0 and not lines[start - 1].strip():
            start -= 1
        return lines[:start] + lines[end:]


__all__ = ["CitationProvenanceMiddleware", "CITED_HOSTS"]
