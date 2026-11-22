"""Normalize documentation results and trim search chunks for model context."""

import re
from collections.abc import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

SNIPPET_CHARS = 300
SEARCH_TOOL_SUFFIX = "search_docs_by_lang_chain"
READ_TOOL_SUFFIX = "query_docs_filesystem_docs_by_lang_chain"


def _is_docs_tool(name: str) -> bool:
    """Match documentation tools with or without their MCP server prefix."""
    return any(
        name == suffix or name.endswith("_" + suffix)
        for suffix in (SEARCH_TOOL_SUFFIX, READ_TOOL_SUFFIX)
    )


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

    def _trim_result(
        self, result: ToolMessage | Command, *, trim_snippets: bool = True
    ) -> ToolMessage | Command:
        """Remove block IDs and optionally trim search snippets in a copied message."""
        if not isinstance(result, ToolMessage):
            return result
        content = result.content
        if isinstance(content, str):
            if trim_snippets:
                content = self._trim_text(content)
        else:
            content = [
                {key: value for key, value in block.items() if key != "id"}
                if isinstance(block, dict)
                else block
                for block in content
            ]
            if trim_snippets:
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
        """Normalize synchronous docs results and trim search snippets."""
        result = handler(request)
        if not _is_docs_tool(request.tool_call["name"]):
            return result
        return self._trim_result(
            result, trim_snippets=request.tool_call["name"].endswith(SEARCH_TOOL_SUFFIX)
        )

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Normalize asynchronous docs results and trim search snippets."""
        result = await handler(request)
        if not _is_docs_tool(request.tool_call["name"]):
            return result
        return self._trim_result(
            result, trim_snippets=request.tool_call["name"].endswith(SEARCH_TOOL_SUFFIX)
        )
