"""Tests for ToolRetryMiddleware silent shell-failure handling."""

import asyncio

from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from src.middleware.tool_retry_middleware import ToolRetryMiddleware


def _make_request(tool_name: str) -> ToolCallRequest:
    request = ToolCallRequest.__new__(ToolCallRequest)
    object.__setattr__(request, "tool_call", {"name": tool_name, "id": "call-1", "args": {}})
    return request


def test_shell_exit_failure_is_rewritten_as_fallback_error():
    """A successful ToolMessage carrying `exit: 1` is rewritten into an error."""
    middleware = ToolRetryMiddleware()
    request = _make_request("query_docs_filesystem_docs_by_lang_chain")
    failure = ToolMessage(
        content="exit: 1",
        status="success",
        name="query_docs_filesystem_docs_by_lang_chain",
        tool_call_id="call-1",
    )

    async def handler(_request):
        return failure

    result = asyncio.run(middleware.awrap_tool_call(request, handler))

    assert isinstance(result, ToolMessage)
    assert "search_docs_by_lang_chain" in result.content
    assert result.content != "exit: 1"


def test_normal_success_message_passes_through_unchanged():
    """A normal successful ToolMessage is returned unchanged."""
    middleware = ToolRetryMiddleware()
    request = _make_request("query_docs_filesystem_docs_by_lang_chain")
    success = ToolMessage(
        content="Found relevant docs about agents.",
        status="success",
        name="query_docs_filesystem_docs_by_lang_chain",
        tool_call_id="call-1",
    )

    async def handler(_request):
        return success

    result = asyncio.run(middleware.awrap_tool_call(request, handler))

    assert result is success
