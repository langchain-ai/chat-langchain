"""Tests that the docs agent's builtin scaffolding tools are stripped."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from langchain_core.tools import tool

from src.middleware.tool_allowlist_middleware import (
    BUILTIN_TOOL_NAMES,
    ToolAllowlistMiddleware,
)


@tool
def search_support_articles(query: str) -> str:
    """Search support articles."""
    return query


def _request_with_tools(tools):
    return SimpleNamespace(
        tools=tools,
        override=lambda **kw: SimpleNamespace(tools=kw["tools"]),
    )


def test_blocked_names_cover_all_scaffolding_builtins():
    assert BUILTIN_TOOL_NAMES == {
        "task",
        "write_todos",
        "write_file",
        "read_file",
        "edit_file",
        "ls",
        "grep",
        "execute",
    }


def test_wrap_model_call_strips_builtins_keeps_docs_tools():
    middleware = ToolAllowlistMiddleware()
    tools = [
        search_support_articles,
        {"name": "task"},
        {"name": "write_todos"},
        {"name": "read_file"},
        {"name": "get_docs"},
    ]
    captured = {}

    def handler(request):
        captured["tools"] = request.tools
        return "ok"

    result = middleware.wrap_model_call(_request_with_tools(tools), handler)

    assert result == "ok"
    names = {getattr(t, "name", None) or t.get("name") for t in captured["tools"]}
    assert names == {"search_support_articles", "get_docs"}
    assert "task" not in names


def test_awrap_model_call_strips_builtins():
    middleware = ToolAllowlistMiddleware()
    tools = [search_support_articles, {"name": "task"}]
    captured = {}

    async def handler(request):
        captured["tools"] = request.tools
        return "ok"

    result = asyncio.run(
        middleware.awrap_model_call(_request_with_tools(tools), handler)
    )

    assert result == "ok"
    names = {getattr(t, "name", None) or t.get("name") for t in captured["tools"]}
    assert names == {"search_support_articles"}


def test_no_override_when_nothing_blocked():
    middleware = ToolAllowlistMiddleware()
    tools = [search_support_articles, {"name": "get_docs"}]

    def handler(request):
        return request.tools

    request = SimpleNamespace(
        tools=tools,
        override=lambda **kw: (_ for _ in ()).throw(
            AssertionError("override should not be called")
        ),
    )

    assert middleware.wrap_model_call(request, handler) == tools
