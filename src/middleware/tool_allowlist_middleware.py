"""Restrict docs_agent to its registered tools by stripping harness builtins.

The ``define_deep_agent`` / ``create_deep_agent`` harness always injects
scaffolding tools (``task``, ``write_todos``, ``write_file``, ``read_file``,
``edit_file``, ``ls``, ``grep``, ``execute``) and has no author-facing flag to
suppress them. The ``task`` builtin in particular lets docs_agent spawn a
general-purpose subagent that runs ungrounded external web searches, breaking
the constraint that answers stay grounded in the registered docs/support tools.

This middleware runs late in the stack (after every tool-injecting middleware)
and removes the injected builtins from the model request, so the model only
ever sees the docs/support tools plus the MCP docs tools.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)

#: Harness scaffolding tools that must never be exposed to the docs agent.
BUILTIN_TOOL_NAMES = frozenset(
    {
        "task",
        "write_todos",
        "write_file",
        "read_file",
        "edit_file",
        "ls",
        "grep",
        "execute",
    }
)


def _tool_name(tool: Any) -> str | None:
    """Return a tool's name whether it is a BaseTool or a dict spec."""
    if isinstance(tool, dict):
        name = tool.get("name")
        return name if isinstance(name, str) else None
    name = getattr(tool, "name", None)
    return name if isinstance(name, str) else None


class ToolAllowlistMiddleware(AgentMiddleware):
    """Strip harness-injected builtin scaffolding tools from the model request."""

    def __init__(self, blocked: frozenset[str] = BUILTIN_TOOL_NAMES) -> None:
        """Store the set of tool names to strip from every model request."""
        super().__init__()
        self._blocked = blocked

    def _filter(self, request: ModelRequest) -> ModelRequest:
        filtered = [t for t in request.tools if _tool_name(t) not in self._blocked]
        if len(filtered) != len(request.tools):
            return request.override(tools=filtered)
        return request

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Remove blocked builtins before the model sees them."""
        return handler(self._filter(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async variant of :meth:`wrap_model_call`."""
        return await handler(self._filter(request))


__all__ = ["ToolAllowlistMiddleware", "BUILTIN_TOOL_NAMES"]
