"""Validate final response links without an extra model/tool round trip."""

import logging
import re
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from src.tools import link_check_tools

logger = logging.getLogger(__name__)

_MARKDOWN_LINK = re.compile(
    r"\[(?P<label>(?:[^\[\]\n]|\[[^\]\n]*\])*)\]\(\s*<?"
    r"(?P<url>https?://(?:[^\s<>()]|\([^\s<>]*\))+)>?"
    r'(?:\s+["\'][^\n]*?["\'])?\s*\)',
    re.IGNORECASE,
)
_PLAIN_URL = re.compile(r'https?://[^\s<>"\']+', re.IGNORECASE)


def _link_spans(text: str) -> list[tuple[int, int, str, str]]:
    """Find URL spans and the text to retain if validation fails."""
    spans = [
        (match.start(), match.end(), match.group("url"), match.group("label"))
        for match in _MARKDOWN_LINK.finditer(text)
    ]
    for match in _PLAIN_URL.finditer(text):
        if any(start <= match.start() < end for start, end, _, _ in spans):
            continue
        url = match.group().rstrip(".,;:!?`]")
        while url.endswith(")") and url.count(")") > url.count("("):
            url = url[:-1]
        start, end = match.start(), match.start() + len(url)
        if start > 0 and text[start - 1] == "<" and text[end : end + 1] == ">":
            start, end = start - 1, end + 1
        spans.append((start, end, url, ""))
    return sorted(spans)


def _text_blocks(content: Any) -> list[str]:
    """Extract text while leaving non-text content blocks alone."""
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        return [
            block if isinstance(block, str) else block["text"]
            for block in content
            if isinstance(block, str)
            or (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            )
        ]
    return []


def _sanitize_text(text: str, valid_urls: set[str]) -> str:
    """Remove invalid URLs, preserving link labels and surrounding prose."""
    for start, end, url, replacement in reversed(_link_spans(text)):
        if url in valid_urls:
            continue
        if (
            not replacement
            and start > 0
            and text[start - 1] == " "
            and text[end : end + 1] == " "
        ):
            end += 1
        text = text[:start] + replacement + text[end:]
    return text


class LinkValidationMiddleware(AgentMiddleware):
    """Sanitize only the final assistant message using the shared link cache."""

    async def aafter_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Validate final URLs and overwrite the message with sanitized content."""
        messages = state.get("messages", [])
        if not messages or not isinstance(messages[-1], AIMessage):
            return None
        message = messages[-1]
        if message.tool_calls:
            return None

        urls = list(
            dict.fromkeys(
                url
                for text in _text_blocks(message.content)
                for _, _, url, _ in _link_spans(text)
            )
        )
        if not urls:
            return None
        pending = [url for url in urls if url not in link_check_tools._cache]
        if pending:
            try:
                results = await link_check_tools._check_urls_async(
                    pending, link_check_tools.DEFAULT_TIMEOUT
                )
                if isinstance(results, (list, tuple)):
                    for result in results:
                        if (
                            isinstance(result, link_check_tools.LinkCheckResult)
                            and isinstance(result.url, str)
                            and result.url in pending
                            and isinstance(result.valid, bool)
                        ):
                            link_check_tools._cache[result.url] = result
            except Exception:
                logger.warning("Final response link validation failed", exc_info=True)

        valid_urls = {
            url
            for url in urls
            if isinstance(
                cached_result := link_check_tools._cache.get(url),
                link_check_tools.LinkCheckResult,
            )
            and cached_result.url == url
            and cached_result.valid is True
        }
        content = message.content
        sanitized: str | list[Any]
        if isinstance(content, str):
            sanitized = _sanitize_text(content, valid_urls)
        else:
            sanitized = [
                _sanitize_text(block, valid_urls)
                if isinstance(block, str)
                else {**block, "text": _sanitize_text(block["text"], valid_urls)}
                if isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
                else block
                for block in content
            ]
        if sanitized == content:
            return None
        return {"messages": [message.model_copy(update={"content": sanitized})]}


__all__ = ["LinkValidationMiddleware"]
