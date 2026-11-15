"""Trim documentation search chunks before they enter model context."""

import re
from collections.abc import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

SNIPPET_CHARS = 300
FULL_RESULTS = 2
# Deep Agents evicts tool text above its default 20,000-token, 4-chars/token limit.
TOOL_RESULT_CHARS = 80_000
SEARCH_TOOL_SUFFIX = "search_docs_by_lang_chain"


def _is_docs_search(name: str) -> bool:
    """Match the docs search tool with or without its MCP server prefix."""
    return name == SEARCH_TOOL_SUFFIX or name.endswith("_" + SEARCH_TOOL_SUFFIX)


class SearchResultTrimMiddleware(AgentMiddleware):
    """Keep top documentation results and bounded snippets with metadata."""

    def _trim_text(self, text: str, result_count: list[int], full_chars: int) -> str:
        """Trim Content sections while counting results across text blocks."""
        sections = re.split(r"(?m)(?=^Title: )", text)
        for index, section in enumerate(sections):
            if not section.startswith("Title: "):
                continue
            limit = full_chars if result_count[0] < FULL_RESULTS else SNIPPET_CHARS
            result_count[0] += 1
            sections[index] = re.sub(
                r"(^Content:[ \t]?)(.*?)(?=^(?:Link|Page|Content):|\Z)",
                lambda match: self._trim_content(match, limit),
                section,
                flags=re.MULTILINE | re.DOTALL,
            )
        return "".join(sections)

    def _trim_content(self, match: re.Match[str], limit: int) -> str:
        """Limit a content body without consuming subsequent metadata lines."""
        header, body = match.groups()
        if len(body) <= limit:
            return match.group(0)
        separator = "\n" if body.endswith("\n") else ""
        return header + body[:limit].rstrip() + "..." + separator

    def _trim_result(self, result: ToolMessage | Command) -> ToolMessage | Command:
        """Preserve top results within the filesystem eviction budget."""
        if not isinstance(result, ToolMessage):
            return result

        def render(full_chars: int) -> ToolMessage:
            result_count = [0]
            content = result.content
            if isinstance(content, str):
                content = self._trim_text(content, result_count, full_chars)
            else:
                content = [
                    {
                        **block,
                        "text": self._trim_text(
                            block["text"], result_count, full_chars
                        ),
                    }
                    if isinstance(block, dict)
                    and block.get("type") == "text"
                    and isinstance(block.get("text"), str)
                    else block
                    for block in content
                ]
            return result.model_copy(update={"content": content})

        def text_size(message: ToolMessage) -> int:
            if isinstance(message.content, str):
                return len(message.content)
            texts = [
                block if isinstance(block, str) else block["text"]
                for block in message.content
                if isinstance(block, str)
                or (
                    isinstance(block, dict)
                    and block.get("type") == "text"
                    and isinstance(block.get("text"), str)
                )
            ]
            return len("\n".join(texts))

        full_chars = text_size(result)
        trimmed = render(full_chars)
        if text_size(trimmed) < TOOL_RESULT_CHARS:
            return trimmed
        low, high = 0, full_chars
        trimmed = render(low)
        while low < high:
            middle = (low + high + 1) // 2
            candidate = render(middle)
            if text_size(candidate) < TOOL_RESULT_CHARS:
                low, trimmed = middle, candidate
            else:
                high = middle - 1
        return trimmed

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Trim synchronous docs search results and pass other tools through."""
        result = handler(request)
        if not _is_docs_search(request.tool_call["name"]):
            return result
        return self._trim_result(result)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Trim asynchronous docs search results and pass other tools through."""
        result = await handler(request)
        if not _is_docs_search(request.tool_call["name"]):
            return result
        return self._trim_result(result)
