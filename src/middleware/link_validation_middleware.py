"""Validate links on the final response without another model call."""

import asyncio
import logging
import re
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from src.tools import link_check_tools

logger = logging.getLogger(__name__)

_MARKDOWN_LINK = re.compile(
    r"!?\[(?P<text>(?:[^\[\]]|\[[^\[\]]*\])*)\]\("
    r"\s*<?(?P<url>https?://(?:[^\s()<>]|\([^\s()]*\))+)>?"
    r'(?:\s+["\'][^"\']*["\'])?\s*\)'
)
_PLAIN_URL = re.compile(r'https?://[^\s<>`"\'\[\]]+')


def _link_spans(text: str) -> list[tuple[int, int, str, str]]:
    """Find link spans and their replacements if validation fails."""
    spans = [
        (match.start(), match.end(), match["url"], match["text"])
        for match in _MARKDOWN_LINK.finditer(text)
    ]
    for match in _PLAIN_URL.finditer(text):
        if any(start <= match.start() < end for start, end, _, _ in spans):
            continue
        url = match.group().rstrip(".,!?:;")
        while url.endswith((")", "}")):
            closing = url[-1]
            opening = "(" if closing == ")" else "{"
            if url.count(closing) <= url.count(opening):
                break
            url = url[:-1]
        start, end = match.start(), match.start() + len(url)
        if start > 0 and end < len(text):
            if (text[start - 1], text[end]) in [("<", ">"), ("(", ")"), ("`", "`")]:
                start -= 1
                end += 1
        spans.append((start, end, url, ""))
    return sorted(spans)


class LinkValidationMiddleware(AgentMiddleware):
    """Sanitize only the final assistant message using the shared link cache."""

    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Validate the final response in synchronous agent runs."""
        return asyncio.run(self.aafter_agent(state, runtime))

    async def aafter_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Remove invalid or unverified links after the agent finishes."""
        messages = state.get("messages", [])
        if not messages or not isinstance(messages[-1], AIMessage):
            return None
        message = messages[-1]
        if message.tool_calls:
            return None

        content = message.content
        blocks = [content] if isinstance(content, str) else content
        texts = [
            block
            if isinstance(block, str)
            else block.get("text", "")
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
            else ""
            for block in blocks
        ]
        spans = [_link_spans(text) for text in texts]
        urls = list(dict.fromkeys(url for links in spans for _, _, url, _ in links))
        if not urls:
            return None

        results = {
            url: link_check_tools._cache[url]
            for url in urls
            if url in link_check_tools._cache
        }
        uncached = [url for url in urls if url not in results]
        if uncached:
            try:
                checked = await link_check_tools._check_urls_async(
                    uncached, link_check_tools.DEFAULT_TIMEOUT
                )
                if isinstance(checked, (list, tuple)):
                    for result in checked:
                        if (
                            isinstance(result, link_check_tools.LinkCheckResult)
                            and isinstance(result.url, str)
                            and result.url in uncached
                            and type(result.valid) is bool
                        ):
                            results[result.url] = result
                            link_check_tools._cache[result.url] = result
            except Exception:
                logger.warning("Final response link validation failed", exc_info=True)

        valid_urls = {
            url
            for url, result in results.items()
            if isinstance(result, link_check_tools.LinkCheckResult)
            and result.url == url
            and result.valid is True
            and result.error is None
        }
        sanitized = []
        for block, text, links in zip(blocks, texts, spans, strict=True):
            for start, end, url, replacement in reversed(links):
                if url not in valid_urls:
                    if not replacement and start > 0 and text[start - 1] in " \t":
                        if end < len(text) and text[end] in " \t":
                            end += 1
                        elif end < len(text) and text[end] in ".,!?;:":
                            start -= 1
                    text = text[:start] + replacement + text[end:]
            if isinstance(block, str):
                sanitized.append(text)
            elif isinstance(block, dict) and block.get("type") == "text":
                sanitized.append({**block, "text": text})
            else:
                sanitized.append(block)
        sanitized_content = sanitized[0] if isinstance(content, str) else sanitized
        if sanitized_content == content:
            return None
        return {"messages": [message.model_copy(update={"content": sanitized_content})]}
