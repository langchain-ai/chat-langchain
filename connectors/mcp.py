"""Managed MCP connector declarations for Chat LangChain."""

from managed_deepagents.connectors import define_mcp_servers

connector = define_mcp_servers(
    prefix_tool_name_with_server_name=False,
    mcp_servers={
        "langchain-docs": {
            "transport": "http",
            "url": "https://docs.langchain.com/mcp",
            # Only append the docs-search tools; never export any tool the docs
            # agent is not scoped for. Matched against the raw MCP tool name
            # before any managed prefix/alias is applied.
            "include_tools": ["search_docs", "fetch_docs"],
        },
    },
)
