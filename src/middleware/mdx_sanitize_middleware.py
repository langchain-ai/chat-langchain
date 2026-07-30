"""Sanitize raw `.mdx` docs tool output before it enters the model context.

The docs filesystem reader is a managed MCP tool (see ``connectors/mcp.py``), so
its result cannot be post-processed inside a tool body; this middleware is the
only boundary where Mintlify/MDX authoring syntax can be removed before the
model can paste it into a user-facing answer.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from src.utils.mdx_sanitize import strip_mdx_artifacts

#: Tools whose results are raw `.mdx` page source rather than prose.
DEFAULT_MDX_TOOL_NAMES = ("query_docs_filesystem_docs_by_lang_chain",)


class MdxSanitizeMiddleware(AgentMiddleware[AgentState]):
    """Strip MDX authoring artifacts from docs-reader tool results."""

    def __init__(self, tool_names: Sequence[str] = DEFAULT_MDX_TOOL_NAMES):
        """Sanitize results for the given tool names."""
        super().__init__()
        self.tool_names = tuple(tool_names)

    def wrap_tool_call(self, request: ToolCallRequest, handler) -> ToolMessage | Command:
        """Sanitize the tool result of a synchronous tool call."""
        return self._sanitize(request, handler(request))

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler
    ) -> ToolMessage | Command:
        """Sanitize the tool result of an async tool call."""
        return self._sanitize(request, await handler(request))

    def _sanitize(
        self, request: ToolCallRequest, result: ToolMessage | Command
    ) -> ToolMessage | Command:
        if request.tool_call.get("name") not in self.tool_names:
            return result
        if not isinstance(result, ToolMessage):
            return result

        sanitized = self._sanitize_content(result.content)
        if sanitized != result.content:
            result.content = sanitized
        return result

    def _sanitize_content(self, content: Any) -> Any:
        if isinstance(content, str):
            return strip_mdx_artifacts(content)

        if not isinstance(content, list):
            return content

        blocks: list[Any] = []
        for block in content:
            if isinstance(block, str):
                blocks.append(strip_mdx_artifacts(block))
            elif (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ):
                blocks.append({**block, "text": strip_mdx_artifacts(block["text"])})
            else:
                blocks.append(block)
        return blocks


__all__ = ["DEFAULT_MDX_TOOL_NAMES", "MdxSanitizeMiddleware"]
