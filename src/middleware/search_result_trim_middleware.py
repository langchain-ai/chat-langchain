"""Trim documentation search chunks before they enter model context."""

import re
from collections.abc import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

SNIPPET_CHARS = 300


class SearchResultTrimMiddleware(AgentMiddleware):
    """Keep documentation search metadata and short content snippets."""

    def _trim_text(self, text: str) -> str:
        """Trim Content sections within Title-delimited search results."""
        sections = re.split(r"(?m)(?=^Title: )", text)
        for index, section in enumerate(sections):
            if not section.startswith("Title: "):
                continue
            sections[index] = re.sub(
                r"(^Content:[ \t]?)(.*?)(?=^(?:Link|Page|Content):|\Z)",
                self._trim_content,
                section,
                flags=re.MULTILINE | re.DOTALL,
            )
        return "".join(sections)

    def _trim_content(self, match: re.Match[str]) -> str:
        """Limit a content body without consuming subsequent metadata lines."""
        header, body = match.groups()
        if len(body) <= SNIPPET_CHARS:
            return match.group(0)
        separator = "\n" if body.endswith("\n") else ""
        return header + body[:SNIPPET_CHARS].rstrip() + "..." + separator

    def _trim_result(self, result: ToolMessage | Command) -> ToolMessage | Command:
        """Copy a tool message with only its text content trimmed."""
        if not isinstance(result, ToolMessage):
            return result
        content = result.content
        if isinstance(content, str):
            content = self._trim_text(content)
        else:
            content = [
                {**block, "text": self._trim_text(block["text"])}
                if isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
                else block
                for block in content
            ]
        return result.model_copy(update={"content": content})

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Trim synchronous docs search results and pass other tools through."""
        result = handler(request)
        if request.tool_call["name"] != "search_docs_by_lang_chain":
            return result
        return self._trim_result(result)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Trim asynchronous docs search results and pass other tools through."""
        result = await handler(request)
        if request.tool_call["name"] != "search_docs_by_lang_chain":
            return result
        return self._trim_result(result)
