"""Tests that filesystem not-found errors name the path the agent requested."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from langchain_core.messages import ToolMessage

from src.middleware.filesystem_paths_middleware import FilesystemPathsMiddleware


def _request(name: str, args: dict) -> SimpleNamespace:
    return SimpleNamespace(
        tool_call={"name": name, "args": args, "id": "call_1"},
        tool=None,
        state={},
        runtime=SimpleNamespace(),
    )


def _run(name: str, args: dict, content: str) -> str:
    middleware = FilesystemPathsMiddleware()
    request = _request(name, args)

    def handler(_request):
        return ToolMessage(content=content, name=name, tool_call_id="call_1")

    result = middleware.wrap_tool_call(request, handler)
    return result.content


def test_read_file_error_keeps_memories_root():
    content = _run(
        "read_file",
        {"file_path": "/memories/wiki/does-not-exist.md", "offset": 0, "limit": 1000},
        "Error: File '/wiki/does-not-exist.md' not found",
    )

    assert "'/memories/wiki/does-not-exist.md'" in content
    assert "'/wiki/does-not-exist.md'" not in content


def test_read_file_error_keeps_system_skills_root():
    content = _run(
        "read_file",
        {"file_path": "/system-skills/some-skill/SKILL.md"},
        "Error: File '/some-skill/SKILL.md' not found",
    )

    assert "'/system-skills/some-skill/SKILL.md'" in content
    assert "'/some-skill/SKILL.md'" not in content


def test_error_enumerates_valid_roots():
    content = _run(
        "read_file",
        {"file_path": "/memories/wiki/does-not-exist.md"},
        "Error: File '/wiki/does-not-exist.md' not found",
    )

    for root in ("/memories/", "/skills/", "/system-skills/", "/tools/"):
        assert root in content


def test_ls_and_glob_errors_keep_requested_path():
    ls_content = _run(
        "ls",
        {"path": "/memories/wiki/missing"},
        "Error: Directory '/wiki/missing' not found",
    )
    assert "'/memories/wiki/missing'" in ls_content

    glob_content = _run(
        "glob",
        {"path": "/system-skills/missing", "pattern": "**/*.md"},
        "Error: Directory '/missing' not found",
    )
    assert "'/system-skills/missing'" in glob_content


def test_already_correct_path_is_left_alone():
    original = "Error: File '/wiki/ats-pipeline/template.md' not found"
    content = _run(
        "read_file",
        {"file_path": "/wiki/ats-pipeline/template.md"},
        original,
    )

    assert content == original


def test_unrelated_tool_output_is_untouched():
    content = _run(
        "read_file",
        {"file_path": "/memories/wiki/index.md"},
        " 1  # Index",
    )

    assert content == " 1  # Index"


def test_async_path_is_repaired():
    middleware = FilesystemPathsMiddleware()
    request = _request("edit_file", {"file_path": "/memories/wiki/missing.md"})

    async def handler(_request):
        return ToolMessage(
            content="Error: File '/wiki/missing.md' not found",
            name="edit_file",
            tool_call_id="call_1",
        )

    result = asyncio.run(middleware.awrap_tool_call(request, handler))

    assert "'/memories/wiki/missing.md'" in result.content
