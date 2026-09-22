"""Tests for the documented tool-call allow-list."""

import asyncio

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from src.middleware.tool_allowlist_middleware import ToolAllowlistMiddleware


def _request(name: str, args: object, call_id: str = "call-1") -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"name": name, "id": call_id, "args": args},
        tool=None,
        state={"messages": [HumanMessage(content="Question")]},
        runtime=None,
    )


def test_allowed_tools_dispatch():
    middleware = ToolAllowlistMiddleware()
    dispatched = []
    allowed_tools = (
        "search_support_articles",
        "get_support_article_content",
        "fetch_langchain_pricing",
        "check_links",
        "search_docs_by_lang_chain",
        "query_docs_filesystem_docs_by_lang_chain",
        "submit_feedback",
        "read_file",
    )

    async def handler(request):
        dispatched.append(request.tool_call["name"])
        return ToolMessage(
            content="ok",
            name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"],
        )

    async def invoke():
        results = []
        for index, tool_name in enumerate(allowed_tools):
            args = {"file_path": f"/large_tool_results/result-{index}.txt"}
            results.append(
                await middleware.awrap_tool_call(_request(tool_name, args), handler)
            )
        return results

    results = asyncio.run(invoke())

    assert dispatched == list(allowed_tools)
    assert all(result.content == "ok" for result in results)


def test_disallowed_tools_are_refused_without_dispatch():
    middleware = ToolAllowlistMiddleware()
    dispatched = []
    disallowed_tools = (
        "ls",
        "write_file",
        "edit_file",
        "delete",
        "glob",
        "grep",
        "execute",
        "task",
        "unknown_tool",
    )

    async def handler(request):
        dispatched.append(request.tool_call["name"])
        return ToolMessage(content="ok")

    async def invoke():
        return [
            await middleware.awrap_tool_call(_request(tool_name, {}), handler)
            for tool_name in disallowed_tools
        ]

    results = asyncio.run(invoke())

    assert dispatched == []
    assert all("not available" in result.content for result in results)


def test_read_file_outside_large_tool_results_is_refused_without_dispatch():
    middleware = ToolAllowlistMiddleware()
    dispatched = []

    async def handler(request):
        dispatched.append(request)
        return ToolMessage(content="ok")

    async def invoke():
        return await middleware.awrap_tool_call(
            _request("read_file", {"file_path": "/large_tool_results/../secret.txt"}),
            handler,
        )

    result = asyncio.run(invoke())

    assert dispatched == []
    assert "/large_tool_results/" in result.content
