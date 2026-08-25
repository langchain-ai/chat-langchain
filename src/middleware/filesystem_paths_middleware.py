"""Report the caller's own path in deep agent filesystem not-found errors."""
import logging
import re
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

logger = logging.getLogger(__name__)

FILESYSTEM_ROOTS = ("/memories/", "/skills/", "/system-skills/", "/tools/")

FILESYSTEM_TOOLS = ("read_file", "write_file", "edit_file", "ls", "glob")

PATH_ARG_NAMES = ("file_path", "path")

NOT_FOUND_PATTERN = re.compile(
    r"^Error: (?P<kind>File|Directory|Path) '(?P<path>[^']*)' not found",
)


class FilesystemPathsMiddleware(AgentMiddleware[AgentState]):
    """Restore the requested path in filesystem tool not-found errors."""

    def _requested_path(self, request: ToolCallRequest) -> str | None:
        args: Any = request.tool_call.get("args") or {}
        if not isinstance(args, dict):
            return None

        for name in PATH_ARG_NAMES:
            value = args.get(name)
            if isinstance(value, str) and value:
                return value

        return None

    def _repair(
        self,
        request: ToolCallRequest,
        result: ToolMessage | Command,
    ) -> ToolMessage | Command:
        if request.tool_call.get("name") not in FILESYSTEM_TOOLS:
            return result

        if not isinstance(result, ToolMessage) or not isinstance(result.content, str):
            return result

        match = NOT_FOUND_PATTERN.match(result.content)
        if match is None:
            return result

        reported = match.group("path")
        requested = self._requested_path(request)
        if not requested or requested == reported:
            return result

        # The filesystem tools strip the mount root (e.g. `/memories`) before
        # formatting the error, so only rewrite the root-stripped suffix case.
        if not requested.endswith(reported):
            return result

        roots = ", ".join(FILESYSTEM_ROOTS)
        content = (
            f"Error: {match.group('kind')} '{requested}' not found. "
            f"Valid filesystem roots: {roots}"
        )
        logger.info(
            "Rewrote %s not-found path %s as requested path %s",
            request.tool_call.get("name"),
            reported,
            requested,
        )
        result.content = content
        return result

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler,
    ) -> ToolMessage | Command:
        """Repair filesystem not-found paths for synchronous tool calls."""
        return self._repair(request, handler(request))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler,
    ) -> ToolMessage | Command:
        """Repair filesystem not-found paths for async tool calls."""
        return self._repair(request, await handler(request))


__all__ = ["FilesystemPathsMiddleware"]
