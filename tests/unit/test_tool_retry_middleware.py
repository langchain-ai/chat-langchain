"""Test per-tool timeout limits and transient error recovery."""

import asyncio
import json

import pytest
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from src.middleware.tool_retry_middleware import ToolRetryMiddleware

FILESYSTEM = "langchainDocs_query_docs_filesystem_docs_by_lang_chain"
SEARCH = "langchainDocs_search_docs_by_lang_chain"


def request_for(name, args):
    return ToolCallRequest(
        tool_call={"name": name, "args": args, "id": "docs-call", "type": "tool_call"},
        tool=None,
        state={},
        runtime=None,
    )


@pytest.mark.parametrize("error", ["Command timed out", "Request TIMEOUT"])
@pytest.mark.parametrize("max_attempts", [1, 3, 5])
def test_filesystem_timeout_returns_guidance_with_capped_attempts(error, max_attempts):
    command = "rg documentation " + "folder/" * 30
    request = request_for(FILESYSTEM, {"command": command})
    calls = []

    async def handler(request):
        calls.append(request)
        raise RuntimeError(error)

    result = asyncio.run(
        ToolRetryMiddleware(max_attempts=max_attempts, initial_delay=0).awrap_tool_call(
            request, handler
        )
    )

    assert len(calls) == min(2, max_attempts)
    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "docs-call"
    assert result.name == FILESYSTEM
    payload = json.loads(result.content)
    assert "10-second limit" in payload["message"]
    assert payload["command"] == command[:120]
    assert "rg" in payload["suggestion"]
    assert SEARCH in payload["suggestion"]


def test_search_timeout_recovers_on_retry():
    request = request_for(SEARCH, {"query": "LangGraph persistence"})
    answer = ToolMessage(content="Found documentation", tool_call_id="docs-call")
    calls = []

    async def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise RuntimeError("Request timed out")
        return answer

    result = asyncio.run(
        ToolRetryMiddleware(initial_delay=0).awrap_tool_call(request, handler)
    )

    assert result is answer
    assert len(calls) == 2


@pytest.mark.parametrize("tool_name", [FILESYSTEM, SEARCH])
@pytest.mark.parametrize("max_attempts", [3, 5])
def test_non_timeout_service_errors_use_full_retry_budget(tool_name, max_attempts):
    request = request_for(tool_name, {"command": "rg docs", "query": "docs"})
    calls = []

    async def handler(request):
        calls.append(request)
        raise RuntimeError("HTTP 503 Service unavailable")

    result = asyncio.run(
        ToolRetryMiddleware(max_attempts=max_attempts, initial_delay=0).awrap_tool_call(
            request, handler
        )
    )

    assert len(calls) == max_attempts
    payload = json.loads(result.content)
    assert payload["error"] == "Tool unavailable"
    assert str(max_attempts) in payload["message"]
