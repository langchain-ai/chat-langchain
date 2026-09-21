"""Restrict documentation filesystem tool calls to read-only corpus paths."""

from __future__ import annotations

import posixpath
import shlex
from pathlib import PurePosixPath
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

DOCS_FILESYSTEM_TOOL = "query_docs_filesystem_docs_by_lang_chain"
_ALLOWED_COMMANDS = {"cat", "head", "rg", "tail"}
_RG_OPTIONS_WITH_VALUES = {"-A", "-B", "-C", "-f", "-g", "-m", "--glob", "--max-count"}
_ALLOWED_ROOTS = ("/oss", "/langsmith", "/api-reference", "/openapi")
_REFUSAL = "Filesystem target not permitted: reads are limited to LangChain documentation files"


def _is_allowed_path(value: str) -> bool:
    if not value.startswith("/"):
        return False
    path = PurePosixPath(value)
    if ".." in path.parts:
        return False
    normalized = posixpath.normpath(value)
    if normalized != value:
        return False
    if path.parent == PurePosixPath("/") and path.suffix == ".mdx":
        return True
    return any(
        normalized.startswith(f"{root}/") for root in _ALLOWED_ROOTS
    )


def _is_filesystem_command_permitted(command: str) -> bool:
    if not command or any(character in command for character in ";|&><`$()\n\r\x00"):
        return False
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return False
    if not tokens or PurePosixPath(tokens[0]).name not in _ALLOWED_COMMANDS:
        return False
    executable = PurePosixPath(tokens[0]).name
    path_tokens = tokens[1:]
    if executable in {"head", "cat", "tail"}:
        path_tokens = [token for token in path_tokens if not token.startswith("-") and not token.isdigit()]
    elif executable == "rg":
        positionals = []
        skip_value = False
        for token in tokens[1:]:
            if skip_value:
                skip_value = False
            elif token in _RG_OPTIONS_WITH_VALUES:
                skip_value = True
            elif not token.startswith("-"):
                positionals.append(token)
        path_tokens = positionals[1:]
    for token in path_tokens:
        if not _is_allowed_path(token):
            return False
    return True


class DocsFilesystemGuardMiddleware(AgentMiddleware):
    """Reject unsafe documentation filesystem commands before dispatch."""

    def _tool_message(self, request: ToolCallRequest) -> ToolMessage:
        tool_call = request.tool_call
        return ToolMessage(
            content=_REFUSAL,
            name=tool_call.get("name", DOCS_FILESYSTEM_TOOL),
            tool_call_id=tool_call.get("id", ""),
            status="error",
        )

    async def awrap_tool_call(self, request: ToolCallRequest, handler) -> ToolMessage | Command:
        """Validate the documentation filesystem command before dispatch."""
        tool_call = request.tool_call
        if tool_call.get("name") != DOCS_FILESYSTEM_TOOL:
            return await handler(request)
        args: Any = tool_call.get("args", {})
        command = args.get("command") if isinstance(args, dict) else None
        if not isinstance(command, str) or not _is_filesystem_command_permitted(command):
            return self._tool_message(request)
        return await handler(request)


__all__ = [
    "DOCS_FILESYSTEM_TOOL",
    "DocsFilesystemGuardMiddleware",
    "_is_filesystem_command_permitted",
]
