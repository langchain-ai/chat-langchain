"""Restrict docs_agent to its intended tools.

``define_deep_agent`` compiles through the deepagents harness, which always
injects a builtin filesystem + planning toolset (``write_todos``, ``write_file``,
``ls``, ``read_file``, ``edit_file``, ``glob``, ``grep``, ``execute``, ``task``)
via ``TodoListMiddleware``/``FilesystemMiddleware``/``SubAgentMiddleware``. The
harness offers no author-facing flag to opt out of those tools, and there is no
user filesystem behind them: when the model calls ``write_file`` the harness
returns a fabricated ``"Updated file ..."`` success. This middleware strips those
tool names from the model request so a stateless docs Q&A agent can never call
them.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse

#: Builtin filesystem + planning tools the docs agent must never expose.
BUILTIN_SCAFFOLDING_TOOLS = frozenset(
    {
        "write_todos",
        "write_file",
        "read_file",
        "edit_file",
        "ls",
        "glob",
        "grep",
        "execute",
        "task",
    }
)


def _tool_name(tool: Any) -> str | None:
    """Return a tool's name whether it is a ``BaseTool`` or a dict spec."""
    if isinstance(tool, dict):
        name = tool.get("name")
        return name if isinstance(name, str) else None
    name = getattr(tool, "name", None)
    return name if isinstance(name, str) else None


class ToolScopeMiddleware(AgentMiddleware):
    """Drop harness-injected filesystem/planning tools before the model sees them."""

    def __init__(self, excluded: frozenset[str] = BUILTIN_SCAFFOLDING_TOOLS) -> None:
        """Store the set of tool names to strip from every model request."""
        super().__init__()
        self._excluded = excluded

    def _filter(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
        filtered = [t for t in request.tools if _tool_name(t) not in self._excluded]
        if len(filtered) != len(request.tools):
            return request.override(tools=filtered)
        return request

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        """Filter excluded tools before the synchronous model call."""
        return handler(self._filter(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        """Filter excluded tools before the async model call."""
        return await handler(self._filter(request))


__all__ = ["BUILTIN_SCAFFOLDING_TOOLS", "ToolScopeMiddleware"]
