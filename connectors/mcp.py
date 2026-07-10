"""Managed MCP connector declarations for Chat LangChain."""

from managed_deepagents.connectors import define_mcp_servers

connector = define_mcp_servers(
    prefix_tool_name_with_server_name=False,
    mcp_servers={
        "langchain-docs": {
            "transport": "http",
            "url": "https://docs.langchain.com/mcp",
        },
    },
)
