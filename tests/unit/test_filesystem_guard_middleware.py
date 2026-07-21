"""Tests confining the docs agent to documentation retrieval."""

from __future__ import annotations

import json
import os

from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

os.environ["USE_LOCAL_PROMPTS"] = "1"

from src.middleware.filesystem_guard_middleware import (
    BLOCKED_TOOLS,
    FilesystemGuardMiddleware,
)


def _request(name: str, args: dict | None = None) -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"name": name, "args": args or {}, "id": "call-1"},
        tool=None,
        state={},
        runtime=None,
    )


def _run(request: ToolCallRequest):
    executed = {"called": False}

    def handler(req):
        executed["called"] = True
        return ToolMessage(content="ok", name=req.tool_call["name"], tool_call_id="call-1")

    result = FilesystemGuardMiddleware().wrap_tool_call(request, handler)
    return result, executed["called"]


def test_manifest_coding_tools_are_all_blocked():
    for name in BLOCKED_TOOLS:
        result, called = _run(_request(name, {"path": "/oss/python/agents.md"}))
        assert not called, f"{name} should never execute"
        assert result.status == "error"


def test_write_file_is_refused_even_on_oss_path():
    result, called = _run(_request("write_file", {"file_path": "/oss/python/x.md"}))
    assert not called
    assert result.status == "error"


def test_user_code_paths_are_refused():
    for tool, key, path in (
        ("read_file", "file_path", "/backend/graphs/agent.py"),
        ("ls", "path", "/home/user/project"),
        ("grep", "pattern", "/"),
        ("read_file", "file_path", "."),
    ):
        result, called = _run(_request(tool, {key: path}))
        assert not called
        assert result.status == "error"
        payload = json.loads(result.content)
        assert "documentation" in payload["suggestion"].lower()


def test_doc_retrieval_tools_pass_through():
    for name in (
        "query_docs_filesystem_docs_by_lang_chain",
        "search_docs_by_lang_chain",
        "search_support_articles",
        "get_support_article_content",
        "check_links",
    ):
        result, called = _run(_request(name, {"query": "middleware"}))
        assert called, f"{name} should execute"
        assert result.status != "error"
