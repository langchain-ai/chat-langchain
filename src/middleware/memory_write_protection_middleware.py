"""Hard write-protect for agent-owned memory/configuration files.

Background
----------
A deep-agent style system that exposes file-writing tools (``edit_file``,
``write_file``) alongside a memory directory containing its own tool surface
(``/memories/tools.json``) and identity (``/memories/AGENTS.md``) is vulnerable
to a privilege-escalation pattern: a non-interactive trigger (e.g. a gmail
cron event) supplies a message that the agent treats as license to extend
its own capabilities, and the agent rewrites its memory to register new MCP
tools without ever asking the user.

This middleware adds a hard, model-independent guard: when a tool call
targets a protected memory path AND the current run is non-interactive
(``source == "trigger"``), the tool call is short-circuited with a synthetic
``ToolMessage`` error. Prompt drift cannot bypass this — the dispatch never
reaches the tool.

The check is opt-in-bypassable via ``skip_memory_write_protection`` in run
metadata, intended for explicit interactive sessions that genuinely need to
edit tool configuration.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

logger = logging.getLogger(__name__)

# File-writing tool names. Any tool call against one of these is inspected.
DEFAULT_WRITE_TOOLS: frozenset[str] = frozenset(
    {
        "edit_file",
        "write_file",
        "str_replace_editor",
        "create_file",
        "patch_file",
    }
)

# Tool-call argument keys that may hold the target path.
PATH_ARG_KEYS: tuple[str, ...] = ("file_path", "path", "filename", "target_path")

# Default protected paths/prefixes under /memories/ that define tool surface
# or agent identity. Anything matching is write-protected on trigger runs.
DEFAULT_PROTECTED_PATHS: frozenset[str] = frozenset(
    {
        "/memories/tools.json",
        "/memories/AGENTS.md",
        "/memories/agents.md",
        "/memories/agent.md",
    }
)

DEFAULT_PROTECTED_PREFIXES: tuple[str, ...] = (
    "/memories/tools/",
    "/memories/agents/",
    "/memories/skills/",
)

DEFAULT_BLOCK_MESSAGE = (
    "Error: Cannot modify protected memory file {path} from a non-interactive "
    "trigger run. Tool-surface and agent-identity files under /memories/ may "
    "only be edited in response to an explicit, interactive user request. "
    "To change tools, ask the user interactively."
)


def _normalize_path(raw: object) -> str | None:
    """Best-effort normalize a path argument to a leading-slash string."""
    if not isinstance(raw, str):
        return None
    stripped = raw.strip()
    if not stripped:
        return None
    # Collapse "./memories/x" or "memories/x" to "/memories/x" so the guard
    # cannot be bypassed by a missing leading slash.
    if stripped.startswith("./"):
        stripped = stripped[1:]
    if not stripped.startswith("/"):
        stripped = "/" + stripped
    return stripped


def _is_protected_path(
    path: str,
    protected_paths: Iterable[str],
    protected_prefixes: Iterable[str],
) -> bool:
    lowered = path.lower()
    if any(lowered == p.lower() for p in protected_paths):
        return True
    return any(lowered.startswith(p.lower()) for p in protected_prefixes)


def _extract_path(tool_args: object) -> str | None:
    if not isinstance(tool_args, dict):
        return None
    for key in PATH_ARG_KEYS:
        if key in tool_args:
            normalized = _normalize_path(tool_args[key])
            if normalized:
                return normalized
    return None


def _run_is_non_interactive(metadata: dict | None) -> bool:
    """A run is non-interactive when its metadata.source is "trigger".

    The hosted runtime sets ``source`` to ``"trigger"`` for cron/event-driven
    runs and ``"interactive"`` (or omits it) for user-initiated chat. We treat
    any explicit ``"trigger"`` value as non-interactive; everything else is
    interactive and unaffected by the guard.
    """
    if not isinstance(metadata, dict):
        return False
    source = metadata.get("source")
    return isinstance(source, str) and source.lower() == "trigger"


def _bypass_requested(metadata: dict | None) -> bool:
    if not isinstance(metadata, dict):
        return False
    flag = metadata.get("skip_memory_write_protection")
    return flag is True


class MemoryWriteProtectionMiddleware(AgentMiddleware[AgentState]):
    """Block writes to agent-owned config files on non-interactive runs.

    Parameters
    ----------
    protected_paths:
        Exact paths (case-insensitive) that may not be written on trigger runs.
    protected_prefixes:
        Path prefixes (case-insensitive) under which all writes are blocked.
    write_tool_names:
        Tool names treated as file writes. Defaults cover common deep-agent
        file tools (``edit_file``, ``write_file``, ``str_replace_editor`` …).
    block_message_template:
        ``str.format`` template used for the synthetic tool error. ``{path}``
        is interpolated.
    """

    def __init__(
        self,
        protected_paths: Iterable[str] = DEFAULT_PROTECTED_PATHS,
        protected_prefixes: Iterable[str] = DEFAULT_PROTECTED_PREFIXES,
        write_tool_names: Iterable[str] = DEFAULT_WRITE_TOOLS,
        block_message_template: str = DEFAULT_BLOCK_MESSAGE,
    ) -> None:
        super().__init__()
        self.protected_paths = frozenset(p.lower() for p in protected_paths)
        self.protected_prefixes = tuple(p.lower() for p in protected_prefixes)
        self.write_tool_names = frozenset(write_tool_names)
        self.block_message_template = block_message_template

    # ------------------------------------------------------------------ helpers
    def _tool_call_metadata(self, request: ToolCallRequest) -> dict | None:
        """Resolve run metadata from the request, tolerating shape variation."""
        # langgraph's ToolCallRequest exposes ``config`` (a RunnableConfig)
        # which carries the run metadata dict the platform attached.
        config = getattr(request, "config", None)
        if isinstance(config, dict):
            metadata = config.get("metadata")
            if isinstance(metadata, dict):
                return metadata

        runtime = getattr(request, "runtime", None)
        if runtime is not None:
            ctx = getattr(runtime, "context", None)
            if isinstance(ctx, dict):
                return ctx
            metadata = getattr(runtime, "metadata", None)
            if isinstance(metadata, dict):
                return metadata
        return None

    def _should_block(
        self,
        tool_name: str,
        tool_args: object,
        metadata: dict | None,
    ) -> str | None:
        """Return the offending path if the call should be blocked, else None."""
        if tool_name not in self.write_tool_names:
            return None
        if not _run_is_non_interactive(metadata):
            return None
        if _bypass_requested(metadata):
            return None
        path = _extract_path(tool_args)
        if not path:
            return None
        if not _is_protected_path(path, self.protected_paths, self.protected_prefixes):
            return None
        return path

    def _block_message(
        self, request: ToolCallRequest, blocked_path: str
    ) -> ToolMessage:
        return ToolMessage(
            content=self.block_message_template.format(path=blocked_path),
            name=request.tool_call.get("name", "unknown_tool"),
            tool_call_id=request.tool_call.get("id", ""),
            status="error",
        )

    # ------------------------------------------------------------------- hooks
    def wrap_tool_call(self, request: ToolCallRequest, handler):
        tool_name = request.tool_call.get("name", "")
        tool_args = request.tool_call.get("args", {})
        metadata = self._tool_call_metadata(request)
        blocked_path = self._should_block(tool_name, tool_args, metadata)
        if blocked_path is not None:
            logger.warning(
                "Blocked %s call against protected memory path %s on "
                "non-interactive trigger run",
                tool_name,
                blocked_path,
            )
            return self._block_message(request, blocked_path)
        return handler(request)

    async def awrap_tool_call(self, request: ToolCallRequest, handler):
        tool_name = request.tool_call.get("name", "")
        tool_args = request.tool_call.get("args", {})
        metadata = self._tool_call_metadata(request)
        blocked_path = self._should_block(tool_name, tool_args, metadata)
        if blocked_path is not None:
            logger.warning(
                "Blocked %s call against protected memory path %s on "
                "non-interactive trigger run",
                tool_name,
                blocked_path,
            )
            return self._block_message(request, blocked_path)
        return await handler(request)


__all__ = ["MemoryWriteProtectionMiddleware"]
