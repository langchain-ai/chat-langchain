"""Validate documentation URLs emitted by the final assistant response."""

import re
from collections.abc import Iterable
from typing import Any
from urllib.parse import urldefrag, urlparse

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.runtime import Runtime

from src.tools.link_check_tools import check_links

URL_PATTERN = re.compile(r"https?://[^\s<>\]\[\)\"']+")
MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
VALID_LINKS_SECTION_PATTERN = re.compile(
    r"(?:^|\n)Valid links:\s*\n(?P<links>(?:\s*-\s+[^\n]+\n?)*)",
    re.IGNORECASE,
)
SUPPORTED_HOSTS = {"docs.langchain.com", "support.langchain.com"}
TRAILING_URL_PUNCTUATION = ".,!?;:"


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        )
    return ""


def _message_urls(message: AIMessage) -> list[str]:
    return [
        url.rstrip(TRAILING_URL_PUNCTUATION)
        for url in URL_PATTERN.findall(_content_text(message.content))
    ]


def _is_supported_url(url: str) -> bool:
    return urlparse(url).netloc.lower().split(":", 1)[0] in SUPPORTED_HOSTS


def _url_bases(urls: Iterable[str]) -> set[str]:
    return {urldefrag(url.rstrip(TRAILING_URL_PUNCTUATION))[0] for url in urls}


def _valid_links_from_tool_message(message: ToolMessage) -> set[str]:
    return _valid_links_from_text(_content_text(message.content))


def _valid_links_from_text(text: str) -> set[str]:
    match = VALID_LINKS_SECTION_PATTERN.search(text)
    if not match:
        return set()
    return {
        url.rstrip(TRAILING_URL_PUNCTUATION)
        for url in URL_PATTERN.findall(match.group("links"))
    }


def _valid_links_from_state(messages: list[Any]) -> set[str]:
    valid_links: set[str] = set()
    for message in messages:
        if isinstance(message, ToolMessage) and message.name == "check_links":
            valid_links.update(_valid_links_from_tool_message(message))
    return valid_links


def _rewrite_content(content: Any, invalid_urls: set[str]) -> Any:
    def rewrite_text(text: str) -> str:
        def rewrite_markdown(match: re.Match[str]) -> str:
            url = match.group(2).rstrip(TRAILING_URL_PUNCTUATION)
            return match.group(1) if url in invalid_urls else match.group(0)

        text = MARKDOWN_LINK_PATTERN.sub(rewrite_markdown, text)

        def rewrite_bare_url(match: re.Match[str]) -> str:
            raw_url = match.group(0)
            url = raw_url.rstrip(TRAILING_URL_PUNCTUATION)
            return raw_url[len(url) :] if url in invalid_urls else raw_url

        return URL_PATTERN.sub(rewrite_bare_url, text)

    if isinstance(content, str):
        return rewrite_text(content)
    if isinstance(content, list):
        rewritten = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                rewritten.append({**block, "text": rewrite_text(block["text"])})
            else:
                rewritten.append(block)
        return rewritten
    return content


class LinkValidationMiddleware(AgentMiddleware[AgentState]):
    """Validate and remove unsupported documentation links from final answers."""

    async def aafter_model(
        self, state: AgentState, runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Validate URLs in the final assistant message."""
        messages = state.get("messages", [])
        if not messages or not isinstance(messages[-1], AIMessage):
            return None

        message = messages[-1]
        if message.tool_calls:
            return None

        emitted_urls = [
            url.rstrip(TRAILING_URL_PUNCTUATION) for url in _message_urls(message)
        ]
        supported_urls = {url for url in emitted_urls if _is_supported_url(url)}
        if not supported_urls:
            return None

        valid_urls = _valid_links_from_state(messages)
        valid_bases = _url_bases(valid_urls)
        unchecked_urls = [
            url for url in supported_urls if urldefrag(url)[0] not in valid_bases
        ]
        if unchecked_urls:
            result = await check_links.ainvoke({"urls": unchecked_urls})
            valid_urls.update(_valid_links_from_text(_content_text(result)))
            valid_bases = _url_bases(valid_urls)

        invalid_urls = {
            url for url in supported_urls if urldefrag(url)[0] not in valid_bases
        }
        if not invalid_urls:
            return None

        updated_message = message.model_copy(
            update={"content": _rewrite_content(message.content, invalid_urls)}
        )
        return {"messages": [updated_message]}
