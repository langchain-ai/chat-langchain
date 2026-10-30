"""Managed MCP declarations for Chat LangChain."""

from managed_deepagents import define_mcp

mcp = define_mcp(
    servers={
        "langchainDocs": {
            "transport": "http",
            "url": "https://docs.langchain.com/mcp",
            "include_tools": [
                "search_docs_by_lang_chain",
                "query_docs_filesystem_docs_by_lang_chain",
            ],
        },
    },
)
