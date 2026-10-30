"""Exercise the managed docs MCP deadline and existing retry path over HTTP."""

import asyncio
import socket
import threading
import time
from copy import deepcopy

import pytest
import uvicorn
from fastmcp import FastMCP
from langgraph.prebuilt.tool_node import ToolCallRequest
from managed_deepagents import define_mcp

from src.middleware.tool_retry_middleware import ToolRetryMiddleware
from tools.mcp import mcp


@pytest.mark.parametrize("stall_first_attempt", [False, True])
def test_docs_search_deadline_and_retry(stall_first_attempt):
    server = FastMCP("docs")
    queries = []

    @server.tool
    async def search_docs_by_lang_chain(query: str) -> str:
        """Search documentation."""
        queries.append(query)
        if stall_first_attempt and len(queries) == 1:
            await asyncio.sleep(60)
        return "Found documentation"

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        http = uvicorn.Server(
            uvicorn.Config(
                server.http_app(), log_level="critical", timeout_graceful_shutdown=1
            )
        )
        thread = threading.Thread(
            target=http.run, kwargs={"sockets": [sock]}, daemon=True
        )
        thread.start()
        try:
            startup_deadline = time.monotonic() + 5
            while not http.started and thread.is_alive():
                assert time.monotonic() < startup_deadline
                time.sleep(0.01)
            assert http.started

            config = deepcopy(mcp.config)
            config["servers"]["langchainDocs"]["url"] = f"http://127.0.0.1:{port}/mcp"
            tool = define_mcp(**config).tools()[0]
            request = ToolCallRequest(
                tool_call={
                    "name": tool.name,
                    "args": {"query": "LangGraph persistence"},
                    "id": "docs-search",
                    "type": "tool_call",
                },
                tool=tool,
                state={},
                runtime=None,
            )
            failed_attempt_durations = []

            async def handler(request):
                started = time.monotonic()
                try:
                    return await tool.ainvoke(request.tool_call)
                except Exception:
                    failed_attempt_durations.append(time.monotonic() - started)
                    raise

            result = asyncio.run(
                ToolRetryMiddleware().awrap_tool_call(request, handler)
            )

            assert "Found documentation" in str(result.content)
            assert result.tool_call_id == "docs-search"
            assert queries == ["LangGraph persistence"] * (
                2 if stall_first_attempt else 1
            )
            if stall_first_attempt:
                assert len(failed_attempt_durations) == 1
                assert 9 <= failed_attempt_durations[0] < 15
            else:
                assert failed_attempt_durations == []
        finally:
            http.should_exit = True
            thread.join(5)
