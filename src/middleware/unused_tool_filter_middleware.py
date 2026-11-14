"""Drop tools the docs agent never uses from every model request.

The managed Deep Agent harness binds its built-in filesystem and subagent
tools (ls, read_file, write_file, edit_file, delete, glob, grep, task) and the
docs MCP server adds submit_feedback. The docs agent reads documentation with
the docs MCP filesystem tool instead, and over 24h of prod traffic (3,564 tool
calls, Nov 13-14) none of these were called once. Their schemas still cost
~2.8k input tokens on every model call (5 calls per answer on average), so we
remove them from the request before it reaches the model.
"""

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware

UNUSED_TOOLS = frozenset(
    {
        "ls",
        "read_file",
        "write_file",
        "edit_file",
        "delete",
        "glob",
        "grep",
        "task",
        "langchainDocs_submit_feedback",
    }
)


def _name(tool: Any) -> str | None:
    if isinstance(tool, dict):
        return tool.get("name") or (tool.get("function") or {}).get("name")
    return getattr(tool, "name", None)


class UnusedToolFilterMiddleware(AgentMiddleware):
    """Remove never-used tools from the tools bound to each model call."""

    def __init__(self, drop: frozenset[str] = UNUSED_TOOLS) -> None:
        super().__init__()
        self.drop = drop

    def _filter(self, request: Any) -> Any:
        tools = list(request.tools or [])
        kept = [t for t in tools if _name(t) not in self.drop]
        if len(kept) == len(tools):
            return request
        return request.override(tools=kept)

    def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        return handler(self._filter(request))

    async def awrap_model_call(
        self, request: Any, handler: Callable[[Any], Awaitable[Any]]
    ) -> Any:
        return await handler(self._filter(request))
