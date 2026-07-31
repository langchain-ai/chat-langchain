"""Tests for re-inlining offloaded tool results and the read-back prompt rule."""

from __future__ import annotations

import os
from types import SimpleNamespace

from langchain_core.messages import AIMessage, ToolMessage

os.environ["USE_LOCAL_PROMPTS"] = "1"

from src.middleware.offloaded_result_middleware import OffloadedToolResultMiddleware
from src.prompts.docs_agent_prompt import docs_agent_prompt

POINTER = (
    "Tool result too large, the result of this tool call call_1 was saved in the "
    "filesystem at this path: /large_tool_results/call_1 You can read the result "
    "from the filesystem by using the read_file tool"
)


def _state(pointer: str, files: dict[str, str]) -> dict:
    return {
        "messages": [
            AIMessage(content=""),
            ToolMessage(content=pointer, tool_call_id="call_1", id="t1"),
        ],
        "files": files,
    }


def test_before_model_reinlines_offloaded_payload():
    middleware = OffloadedToolResultMiddleware()
    state = _state(POINTER, {"/large_tool_results/call_1": "docs payload"})

    update = middleware.before_model(state, runtime=SimpleNamespace())

    assert update is not None
    assert update["messages"][0].id == "t1"
    assert update["messages"][0].content == "docs payload"


def test_before_model_truncates_with_continuation_pointer():
    middleware = OffloadedToolResultMiddleware(max_inline_chars=10)
    state = _state(POINTER, {"/large_tool_results/call_1": "x" * 50})

    update = middleware.before_model(state, runtime=SimpleNamespace())

    content = update["messages"][0].content
    assert content.startswith("x" * 10)
    assert "/large_tool_results/call_1" in content


def test_before_model_noop_without_pointer():
    middleware = OffloadedToolResultMiddleware()
    state = _state("real content", {})

    assert middleware.before_model(state, runtime=SimpleNamespace()) is None


def test_before_model_leaves_pointer_when_payload_missing():
    middleware = OffloadedToolResultMiddleware()
    state = _state(POINTER, {})

    assert middleware.before_model(state, runtime=SimpleNamespace()) is None


def test_prompt_documents_read_file_and_offload_rule():
    assert "read_file" in docs_agent_prompt
    assert "/large_tool_results/" in docs_agent_prompt
    assert "POINTER, NOT CONTENT" in docs_agent_prompt
