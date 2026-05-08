# MCP (Model Context Protocol) tools loader for official LangChain docs.
import asyncio
import logging

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger(__name__)

MCP_SERVER_URL = "https://docs.langchain.com/mcp"
MCP_SERVER_NAME = "langchain-docs"
MCP_TRANSPORT = "streamable_http"


def _get_mcp_tools() -> list[BaseTool]:
    async def _fetch_tools() -> list[BaseTool]:
        client = MultiServerMCPClient(
            {MCP_SERVER_NAME: {"url": MCP_SERVER_URL, "transport": MCP_TRANSPORT}}
        )
        return await client.get_tools()

    try:
        tools = asyncio.run(_fetch_tools())
        logger.info(f"MCP docs tools loaded: {[tool.name for tool in tools]}")
        return tools
    except Exception as e:
        logger.error(f"Failed to load MCP docs tools: {e}")
        return []


mcp_docs_tools: list[BaseTool] = _get_mcp_tools()

logger.info(f"MCP docs tools module initialized with {len(mcp_docs_tools)} tools")
