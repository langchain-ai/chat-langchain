"""Restrict documentation filesystem tool calls to the docs corpus."""

import posixpath
import shlex
from collections.abc import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

DOCS_FILESYSTEM_TOOL = "query_docs_filesystem_docs_by_lang_chain"
DOCS_CORPUS_ROOTS = ("api-reference", "langsmith", "openapi", "oss")
LARGE_TOOL_RESULTS_ROOT = "/large_tool_results"


class FilesystemPathGuardMiddleware(AgentMiddleware[AgentState]):
    """Reject documentation tool commands that target paths outside the corpus."""

    def _tool_message(self, request: ToolCallRequest, content: str) -> ToolMessage:
        tool_call = request.tool_call
        return ToolMessage(
            content=content,
            name=tool_call.get("name", DOCS_FILESYSTEM_TOOL),
            tool_call_id=tool_call.get("id", ""),
            status="error",
        )

    def _path_is_allowed(self, value: str) -> bool:
        if value.startswith("~"):
            return False

        value = value.strip("\"'()[]{};,")
        if not value or value in {".", "/"}:
            return True
        if ".." in value.split("/"):
            return False

        normalized = posixpath.normpath(value)
        parts = normalized.split("/")
        if ".." in parts or "~" in parts:
            return False

        if normalized.startswith("/"):
            if normalized == LARGE_TOOL_RESULTS_ROOT or normalized.startswith(
                f"{LARGE_TOOL_RESULTS_ROOT}/"
            ):
                return True
            if any(
                normalized == f"/{root}" or normalized.startswith(f"/{root}/")
                for root in DOCS_CORPUS_ROOTS
            ):
                return True
            return normalized.count("/") == 1 and normalized.endswith(".mdx")

        if normalized.endswith(".mdx") and "/" not in normalized:
            return True
        return any(
            normalized == root or normalized.startswith(f"{root}/")
            for root in DOCS_CORPUS_ROOTS
        )

    def _disallowed_target(self, command: str) -> str | None:
        try:
            tokens = shlex.split(command)
        except ValueError:
            return "an invalid shell command"

        for token in tokens:
            candidates = [token]
            if "=" in token:
                candidates.append(token.split("=", 1)[1])
            for candidate in candidates:
                candidate = candidate.strip("\"'()[]{};,")
                if candidate.startswith("-") and "/" not in candidate:
                    continue
                if not (
                    candidate.startswith(("/", "~", "."))
                    or "/" in candidate
                    or candidate.endswith(".mdx")
                ):
                    continue
                if not self._path_is_allowed(candidate):
                    return candidate
        return None

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Validate documentation filesystem targets before dispatch."""
        if request.tool_call.get("name") != DOCS_FILESYSTEM_TOOL:
            return await handler(request)

        args = request.tool_call.get("args", {})
        command = args.get("command") if isinstance(args, dict) else None
        if not isinstance(command, str):
            return self._tool_message(
                request,
                "Rejected documentation filesystem call: command must be a string.",
            )

        target = self._disallowed_target(command)
        if target is not None:
            return self._tool_message(
                request,
                "Rejected documentation filesystem call: path targets must remain "
                f"inside the documentation corpus; disallowed target: {target}",
            )
        return await handler(request)


__all__ = ["FilesystemPathGuardMiddleware"]
