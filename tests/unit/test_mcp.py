"""Tests for the documentation MCP tool boundary."""

import asyncio

from fastmcp import Client, FastMCP
from managed_deepagents._connectors.mcp import load as mcp_load

from tools.mcp import mcp


def test_docs_mcp_exposes_read_tools_without_feedback(monkeypatch):
    server = FastMCP("docs")

    @server.tool
    def search_docs_by_lang_chain(query: str) -> str:
        return f"Found docs for {query}"

    @server.tool
    def query_docs_filesystem_docs_by_lang_chain(command: str) -> str:
        return f"Read docs with {command}"

    @server.tool
    def submit_feedback(feedback: str) -> str:
        raise AssertionError("Feedback must not be submitted")

    async def client_factory(*args):
        return Client(server)

    monkeypatch.setattr(mcp_load, "_client", client_factory)

    tools = {tool.name: tool for tool in mcp.tools()}
    assert set(tools) == {
        "langchainDocs_search_docs_by_lang_chain",
        "langchainDocs_query_docs_filesystem_docs_by_lang_chain",
    }
    search_result = asyncio.run(
        tools["langchainDocs_search_docs_by_lang_chain"].ainvoke({"query": "agents"})
    )
    read_result = asyncio.run(
        tools["langchainDocs_query_docs_filesystem_docs_by_lang_chain"].ainvoke(
            {"command": "head -80 /agents.mdx"}
        )
    )
    assert search_result[0]["text"] == "Found docs for agents"
    assert read_result[0]["text"] == "Read docs with head -80 /agents.mdx"
