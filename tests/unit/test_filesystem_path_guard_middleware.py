"""Tests for documentation filesystem path restrictions."""

import asyncio
from types import SimpleNamespace

from langchain_core.messages import ToolMessage

from src.middleware.filesystem_path_guard_middleware import (
    FilesystemPathGuardMiddleware,
)


def _request(command: str):
    return SimpleNamespace(
        tool_call={
            "name": "query_docs_filesystem_docs_by_lang_chain",
            "args": {"command": command},
            "id": "call-1",
        }
    )


def test_allows_docs_corpus_path():
    calls = []

    async def handler(request):
        calls.append(request)
        return ToolMessage(
            content="docs",
            name=request.tool_call["name"],
            tool_call_id="call-1",
        )

    result = asyncio.run(
        FilesystemPathGuardMiddleware().awrap_tool_call(
            _request("head -5 /oss/python/langgraph/streaming.mdx"), handler
        )
    )

    assert result.content == "docs"
    assert len(calls) == 1


def test_rejects_system_path_without_invoking_tool():
    calls = []

    async def handler(request):
        calls.append(request)
        raise AssertionError("rejected call reached the underlying tool")

    result = asyncio.run(
        FilesystemPathGuardMiddleware().awrap_tool_call(
            _request("head -5 /etc/hostname"), handler
        )
    )

    assert result.status == "error"
    assert "/etc/hostname" in result.content
    assert calls == []


def test_rejects_parent_traversal_without_invoking_tool():
    calls = []

    async def handler(request):
        calls.append(request)
        raise AssertionError("rejected call reached the underlying tool")

    result = asyncio.run(
        FilesystemPathGuardMiddleware().awrap_tool_call(
            _request("head -5 /oss/../etc/hostname"), handler
        )
    )

    assert result.status == "error"
    assert "hostname" in result.content
    assert calls == []
