"""Unit tests for MemoryWriteProtectionMiddleware.

Simulates the gmail-event cron payload from the reference incident and
asserts that ``edit_file`` against ``/memories/tools.json`` is short-
circuited rather than dispatched to the underlying handler.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from src.middleware.memory_write_protection_middleware import (
    DEFAULT_PROTECTED_PATHS,
    MemoryWriteProtectionMiddleware,
)


# ---------------------------------------------------------------------------
# Lightweight stand-ins for ToolCallRequest so the tests don't depend on
# langgraph's internal dataclass shape.
# ---------------------------------------------------------------------------


@dataclass
class _FakeRequest:
    tool_call: dict[str, Any]
    config: dict[str, Any] = field(default_factory=dict)


async def _async_handler_should_not_run(_request):  # pragma: no cover - asserts
    raise AssertionError(
        "Handler should not be invoked when the call is short-circuited"
    )


def _sync_handler_should_not_run(_request):  # pragma: no cover - asserts
    raise AssertionError(
        "Handler should not be invoked when the call is short-circuited"
    )


def _make_request(
    *,
    tool_name: str = "edit_file",
    path: str = "/memories/tools.json",
    source: str | None = "trigger",
    skip: bool | None = None,
    extra_args: dict[str, Any] | None = None,
) -> _FakeRequest:
    metadata: dict[str, Any] = {}
    if source is not None:
        metadata["source"] = source
    if skip is not None:
        metadata["skip_memory_write_protection"] = skip
    args: dict[str, Any] = {"file_path": path}
    if extra_args:
        args.update(extra_args)
    return _FakeRequest(
        tool_call={
            "name": tool_name,
            "id": "call-1",
            "args": args,
        },
        config={"metadata": metadata},
    )


# ---------------------------------------------------------------------------
# Blocking cases
# ---------------------------------------------------------------------------


def test_blocks_edit_file_against_tools_json_on_trigger_run():
    """The reference incident: gmail-cron run editing /memories/tools.json."""
    mw = MemoryWriteProtectionMiddleware()
    req = _make_request(
        tool_name="edit_file",
        path="/memories/tools.json",
        source="trigger",
    )

    result = asyncio.run(mw.awrap_tool_call(req, _async_handler_should_not_run))

    assert "Cannot modify protected memory file" in result.content
    assert "/memories/tools.json" in result.content
    assert result.tool_call_id == "call-1"
    assert result.name == "edit_file"
    assert getattr(result, "status", None) == "error"


def test_blocks_write_file_against_agents_md_on_trigger_run():
    mw = MemoryWriteProtectionMiddleware()
    req = _make_request(
        tool_name="write_file",
        path="/memories/AGENTS.md",
        source="trigger",
    )

    result = asyncio.run(mw.awrap_tool_call(req, _async_handler_should_not_run))

    assert "Cannot modify protected memory file" in result.content
    assert "/memories/AGENTS.md" in result.content


def test_blocks_case_insensitive_path():
    """Path comparison is case-insensitive to defeat trivial bypass."""
    mw = MemoryWriteProtectionMiddleware()
    req = _make_request(path="/Memories/Tools.JSON", source="trigger")
    result = asyncio.run(mw.awrap_tool_call(req, _async_handler_should_not_run))
    assert "Cannot modify protected memory file" in result.content


def test_blocks_path_without_leading_slash():
    """Missing leading slash is normalized; the guard still fires."""
    mw = MemoryWriteProtectionMiddleware()
    req = _make_request(path="memories/tools.json", source="trigger")
    result = asyncio.run(mw.awrap_tool_call(req, _async_handler_should_not_run))
    assert "Cannot modify protected memory file" in result.content


def test_blocks_protected_prefix_paths():
    """Writes under protected prefixes are also blocked, not just the named files."""
    mw = MemoryWriteProtectionMiddleware()
    req = _make_request(
        path="/memories/tools/linear.json",
        source="trigger",
    )
    result = asyncio.run(mw.awrap_tool_call(req, _async_handler_should_not_run))
    assert "Cannot modify protected memory file" in result.content


def test_sync_wrap_blocks_too():
    mw = MemoryWriteProtectionMiddleware()
    req = _make_request(source="trigger")
    result = mw.wrap_tool_call(req, _sync_handler_should_not_run)
    assert "Cannot modify protected memory file" in result.content


# ---------------------------------------------------------------------------
# Allow cases — make sure the guard does NOT interfere with legitimate flows.
# ---------------------------------------------------------------------------


def test_allows_interactive_run_to_edit_tools_json():
    """An explicit user (interactive) run is allowed to edit tools.json."""
    mw = MemoryWriteProtectionMiddleware()
    req = _make_request(source="interactive")

    called: dict[str, bool] = {"yes": False}

    async def handler(_request):
        called["yes"] = True
        return "ok"

    result = asyncio.run(mw.awrap_tool_call(req, handler))
    assert called["yes"] is True
    assert result == "ok"


def test_allows_missing_source_metadata():
    """If no source is set, treat as interactive (don't block)."""
    mw = MemoryWriteProtectionMiddleware()
    req = _make_request(source=None)

    async def handler(_request):
        return "ok"

    result = asyncio.run(mw.awrap_tool_call(req, handler))
    assert result == "ok"


def test_allows_when_skip_flag_explicitly_true():
    """Explicit opt-out via skip_memory_write_protection bypasses the guard."""
    mw = MemoryWriteProtectionMiddleware()
    req = _make_request(source="trigger", skip=True)

    async def handler(_request):
        return "ok"

    result = asyncio.run(mw.awrap_tool_call(req, handler))
    assert result == "ok"


def test_allows_writes_to_unprotected_paths_on_trigger_run():
    """A cron run editing a user-data file (not tool surface) is fine."""
    mw = MemoryWriteProtectionMiddleware()
    req = _make_request(path="/memories/notes/2024.md", source="trigger")

    async def handler(_request):
        return "ok"

    result = asyncio.run(mw.awrap_tool_call(req, handler))
    assert result == "ok"


def test_allows_non_write_tools():
    """Read tools, MCP calls, etc. are never blocked even on trigger runs."""
    mw = MemoryWriteProtectionMiddleware()
    req = _make_request(
        tool_name="read_file",
        path="/memories/tools.json",
        source="trigger",
    )

    async def handler(_request):
        return "ok"

    result = asyncio.run(mw.awrap_tool_call(req, handler))
    assert result == "ok"


def test_default_protected_paths_cover_incident_files():
    """Regression: both files from the incident must be in defaults."""
    lowered = {p.lower() for p in DEFAULT_PROTECTED_PATHS}
    assert "/memories/tools.json" in lowered
    assert "/memories/agents.md" in lowered


def test_gmail_cron_payload_simulation():
    """End-to-end-ish: simulate the four linear-tool adds from the incident.

    The reference incident issued four ``edit_file`` calls against
    ``/memories/tools.json`` to add Linear MCP tools. Each must be blocked
    and the underlying handler must never run.
    """
    mw = MemoryWriteProtectionMiddleware()

    tool_additions = [
        "linear_list_issues",
        "linear_get_issue",
        "linear_list_teams",
        "linear_list_team_members",
    ]

    for added_tool in tool_additions:
        req = _FakeRequest(
            tool_call={
                "name": "edit_file",
                "id": f"call-{added_tool}",
                "args": {
                    "file_path": "/memories/tools.json",
                    "old_string": "{",
                    "new_string": (
                        '{ "' + added_tool + '": { "server": "linear" }, '
                    ),
                },
            },
            config={
                "metadata": {
                    "source": "trigger",
                    "skip_memory_write_protection": False,
                }
            },
        )

        async def handler_must_not_run(_request):  # pragma: no cover - asserts
            raise AssertionError(
                f"edit_file for {added_tool} should have been blocked"
            )

        result = asyncio.run(mw.awrap_tool_call(req, handler_must_not_run))
        assert "Cannot modify protected memory file" in result.content
        assert "/memories/tools.json" in result.content


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
