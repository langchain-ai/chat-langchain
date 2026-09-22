"""Restrict tool calls to the documented agent surface."""

import posixpath
from collections.abc import Awaitable, Callable, Mapping

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

_ALLOWED_TOOLS = frozenset(
    {
        "search_support_articles",
        "get_support_article_content",
        "fetch_langchain_pricing",
        "check_links",
        "search_docs_by_lang_chain",
        "query_docs_filesystem_docs_by_lang_chain",
        "submit_feedback",
        "read_file",
    }
)
_LARGE_TOOL_RESULTS_PREFIX = "/large_tool_results/"


class ToolAllowlistMiddleware(AgentMiddleware):
    """Refuse tool calls outside the documented agent surface."""

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Allow documented tools and the scoped result-file reader."""
        tool_call = request.tool_call
        tool_name = tool_call.get("name")
        if not isinstance(tool_name, str) or tool_name not in _ALLOWED_TOOLS:
            return self._refusal(request, f"Tool {tool_name!r} is not available.")
        if tool_name == "read_file" and not self._is_allowed_read_file(
            tool_call.get("args")
        ):
            return self._refusal(
                request,
                "read_file is only available for files under /large_tool_results/.",
            )
        return await handler(request)

    def _is_allowed_read_file(self, args: object) -> bool:
        if not isinstance(args, Mapping):
            return False
        file_path = args.get("file_path")
        if not isinstance(file_path, str):
            return False
        normalized_path = posixpath.normpath(file_path)
        return normalized_path.startswith(_LARGE_TOOL_RESULTS_PREFIX)

    def _refusal(self, request: ToolCallRequest, content: str) -> ToolMessage:
        return ToolMessage(
            content=content,
            name=request.tool_call.get("name", "unknown_tool"),
            tool_call_id=request.tool_call.get("id", ""),
        )
