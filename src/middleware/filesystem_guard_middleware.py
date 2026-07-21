"""Confine the docs agent to documentation retrieval.

The Managed Deep Agent runtime injects general coding/orchestration tools
(``ls``, ``grep``, ``read_file``, ``write_file``, ``write_todos``, ``task``)
alongside the doc-retrieval tools declared in ``agent.py``. Left unguarded, the
docs agent uses them to browse and even patch a user's own codebase instead of
answering from LangChain documentation. This middleware intercepts those tool
calls before execution: blocked tools are refused outright, and any filesystem
tool that survives may only touch the documentation roots under ``/oss/``.
"""

from __future__ import annotations

import json
import logging
import posixpath
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

logger = logging.getLogger(__name__)

#: Coding/orchestration tools the docs agent must never use. They let the agent
#: read and modify the user's own source instead of retrieving documentation.
BLOCKED_TOOLS = frozenset(
    {"ls", "grep", "read_file", "write_file", "write_todos", "task"}
)

#: Documentation roots the agent is allowed to touch if a filesystem tool is
#: permitted to run. Everything else (``/backend/``, ``/home/``, ``/``, ``.``)
#: is off-limits.
ALLOWED_PATH_PREFIXES = ("/oss/python/", "/oss/javascript/")

#: Tool-call argument keys that may carry a filesystem path or shell command.
PATH_ARG_KEYS = ("path", "file_path", "command", "pattern")

_REDIRECT = (
    "Use the documentation-retrieval tools (search_docs_by_lang_chain, "
    "query_docs_filesystem_docs_by_lang_chain, search_support_articles, "
    "get_support_article_content) to answer from LangChain documentation. Never "
    "read, analyze, or modify the user's own source code or files."
)


class FilesystemGuardMiddleware(AgentMiddleware[AgentState]):
    """Refuse coding tools and confine filesystem tools to the docs roots."""

    def _refuse(self, request: ToolCallRequest, message: str) -> ToolMessage:
        payload = {
            "error": "Operation not permitted",
            "message": message,
            "suggestion": _REDIRECT,
        }
        return ToolMessage(
            content=json.dumps(payload),
            name=request.tool_call.get("name", "unknown_tool"),
            tool_call_id=request.tool_call.get("id", ""),
            status="error",
        )

    def _is_allowed_path(self, value: str) -> bool:
        candidate = value.strip()
        if not candidate.startswith("/"):
            return False
        normalized = posixpath.normpath(candidate)
        if not normalized.endswith("/"):
            normalized += "/"
        return normalized.startswith(ALLOWED_PATH_PREFIXES)

    def _out_of_scope_arg(self, args: dict[str, Any]) -> str | None:
        for key in PATH_ARG_KEYS:
            value = args.get(key)
            if isinstance(value, str) and value and not self._is_allowed_path(value):
                return key
        return None

    def _screen(self, request: ToolCallRequest) -> ToolMessage | None:
        name = request.tool_call.get("name", "")
        if name in BLOCKED_TOOLS:
            logger.warning("Refused blocked tool %s for docs agent", name)
            return self._refuse(
                request,
                f"{name} is disabled for the documentation assistant.",
            )
        args = request.tool_call.get("args") or {}
        if isinstance(args, dict) and (bad_key := self._out_of_scope_arg(args)):
            logger.warning(
                "Refused %s: %s outside documentation roots", name, bad_key
            )
            return self._refuse(
                request,
                f"{name} may only operate under {', '.join(ALLOWED_PATH_PREFIXES)}; "
                f"the {bad_key} argument resolves outside the documentation roots.",
            )
        return None

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler,
    ) -> ToolMessage | Command:
        """Refuse out-of-scope tool calls before they execute."""
        refusal = self._screen(request)
        if refusal is not None:
            return refusal
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler,
    ) -> ToolMessage | Command:
        """Async variant of :meth:`wrap_tool_call`."""
        refusal = self._screen(request)
        if refusal is not None:
            return refusal
        return await handler(request)


__all__ = [
    "ALLOWED_PATH_PREFIXES",
    "BLOCKED_TOOLS",
    "FilesystemGuardMiddleware",
]
