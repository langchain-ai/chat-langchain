# MCP (Model Context Protocol) tools loader for official LangChain docs.
import asyncio
import logging

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger(__name__)

MCP_SERVER_URL = "https://docs.langchain.com/mcp"
MCP_SERVER_NAME = "langchain-docs"
MCP_TRANSPORT = "streamable_http"

# Only these docs-search tools are exposed to the agent. The MCP export can also
# spread builtin filesystem scaffolding (grep, ls, ...) which the docs agent must
# never receive; leaking it caused runaway grep loops with no final answer.
DOCS_TOOL_ALLOWLIST = {
    "search_docs_by_lang_chain",
    "query_docs_filesystem_docs_by_lang_chain",
}
BUILTIN_SCAFFOLDING_TOOLS = {
    "grep",
    "ls",
    "read_file",
    "write_file",
    "write_todos",
    "task",
}


def _get_mcp_tools() -> list[BaseTool]:
    async def _fetch_tools() -> list[BaseTool]:
        client = MultiServerMCPClient(
            {MCP_SERVER_NAME: {"url": MCP_SERVER_URL, "transport": MCP_TRANSPORT}}
        )
        return await client.get_tools()

    try:
        tools = asyncio.run(_fetch_tools())
        filtered = [
            t
            for t in tools
            if t.name in DOCS_TOOL_ALLOWLIST
            and t.name not in BUILTIN_SCAFFOLDING_TOOLS
        ]
        dropped = [t.name for t in tools if t not in filtered]
        logger.info(
            f"MCP docs tools kept: {[t.name for t in filtered]}; dropped: {dropped}"
        )
        return filtered
    except Exception as e:
        logger.error(f"Failed to load MCP docs tools: {e}")
        return []


mcp_docs_tools: list[BaseTool] = _get_mcp_tools()

logger.info(f"MCP docs tools module initialized with {len(mcp_docs_tools)} tools")
