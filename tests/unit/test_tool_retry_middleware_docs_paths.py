"""Tests for docs-filesystem invalid path/command detection in ToolRetryMiddleware."""

from __future__ import annotations

import os

import anyio
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

os.environ["USE_LOCAL_PROMPTS"] = "1"

from src.middleware.tool_retry_middleware import (
    DOCS_FILESYSTEM_TOOL,
    ToolRetryMiddleware,
)


def _request(tool_name: str) -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={
            "name": tool_name,
            "args": {"command": "head -120 /oss/python/integrations/chat/deepseek.mdx"},
            "id": "call-1",
            "type": "tool_call",
        },
        tool=None,
        state={},
        runtime=None,
    )


def _run(tool_name: str, content: str) -> ToolMessage:
    middleware = ToolRetryMiddleware()
    request = _request(tool_name)
    result = ToolMessage(content=content, name=tool_name, tool_call_id="call-1")

    async def handler(_request):
        return result

    return anyio.run(middleware.awrap_tool_call, request, handler)


def test_missing_path_is_flagged_for_the_model():
    output = _run(
        DOCS_FILESYSTEM_TOOL,
        "exit: 1\n--- stderr ---\nhead: /oss/python/integrations/chat/deepseek.mdx: "
        "No such file or directory",
    )

    assert output.content.startswith("ERROR_DOCS_PATH_INVALID:")
    assert "search_docs_by_lang_chain" in output.content


def test_permission_denied_is_flagged_for_the_model():
    output = _run(
        DOCS_FILESYSTEM_TOOL,
        "exit: 126\n--- stderr ---\noffset=120: Permission denied",
    )

    assert output.content.startswith("ERROR_DOCS_PATH_INVALID:")


def test_successful_read_is_returned_unchanged():
    content = "exit: 0\n--- stdout ---\n# DeepSeek\nSome docs body."

    assert _run(DOCS_FILESYSTEM_TOOL, content).content == content


def test_ripgrep_no_match_is_not_flagged():
    content = "exit: 1"

    assert _run(DOCS_FILESYSTEM_TOOL, content).content == content


def test_other_tools_are_never_rewritten():
    content = "exit: 1\n--- stderr ---\nNo such file or directory"

    assert _run("search_docs_by_lang_chain", content).content == content
