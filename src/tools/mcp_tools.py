# MCP (Model Context Protocol) tools loader for official LangChain docs.
import asyncio
import logging

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger(__name__)

MCP_SERVER_URL = "https://docs.langchain.com/mcp"
MCP_SERVER_NAME = "langchain-docs"
MCP_TRANSPORT = "streamable_http"

# Tools advertised by the upstream MCP server that are not actually registered
# server-side: every invocation returns MCP error -32602 "Tool not found". Drop
# them from the bound toolset so the model stops trying to call them.
UNREGISTERED_TOOL_NAMES = frozenset({"query_docs_filesystem_docs_by_lang_chain"})


def _surface_empty_result(tool: BaseTool) -> BaseTool:
    """Wrap a tool so empty MCP results surface a readable error, not silence."""
    original_arun = tool._arun

    async def _arun_with_error_surfacing(*args, **kwargs):
        result = await original_arun(*args, **kwargs)
        # langchain_mcp_adapters renders an upstream MCP error (e.g. -32602
        # "Tool not found") as empty content rather than raising, which the
        # model cannot distinguish from a legitimate "no matches" result.
        if isinstance(result, ToolMessage):
            if not result.content:
                result.content = f"tool unavailable: {tool.name}"
            return result
        if result in ([], "", None):
            return f"tool unavailable: {tool.name}"
        return result

    tool._arun = _arun_with_error_surfacing
    return tool


def _get_mcp_tools() -> list[BaseTool]:
    async def _fetch_tools() -> list[BaseTool]:
        client = MultiServerMCPClient(
            {MCP_SERVER_NAME: {"url": MCP_SERVER_URL, "transport": MCP_TRANSPORT}}
        )
        return await client.get_tools()

    try:
        tools = asyncio.run(_fetch_tools())
        dropped = [t.name for t in tools if t.name in UNREGISTERED_TOOL_NAMES]
        if dropped:
            logger.warning(f"Dropping unregistered MCP docs tools: {dropped}")
        tools = [
            _surface_empty_result(t)
            for t in tools
            if t.name not in UNREGISTERED_TOOL_NAMES
        ]
        logger.info(f"MCP docs tools loaded: {[tool.name for tool in tools]}")
        return tools
    except Exception as e:
        logger.error(f"Failed to load MCP docs tools: {e}")
        return []


mcp_docs_tools: list[BaseTool] = _get_mcp_tools()

logger.info(f"MCP docs tools module initialized with {len(mcp_docs_tools)} tools")
