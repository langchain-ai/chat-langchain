"""Managed MCP connector declarations for Chat LangChain."""

from managed_deepagents.connectors import define_mcp_servers

connector = define_mcp_servers(
    prefix_tool_name_with_server_name=False,
    mcp_servers={
        "langchain-docs": {
            "transport": "http",
            "url": "https://docs.langchain.com/mcp",
            # The docs agent is a public-facing read-only research agent. It must
            # never expose `submit_feedback`, a write tool whose arguments the
            # model would otherwise author itself.
            "exclude_tools": ["submit_feedback"],
        },
    },
)
