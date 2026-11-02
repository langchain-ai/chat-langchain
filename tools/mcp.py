"""Managed MCP declarations for Chat LangChain."""

import os

from managed_deepagents import define_mcp

mcp = define_mcp(
    servers={
        "langchainDocs": {
            "transport": "http",
            "url": "https://docs.langchain.com/mcp",
            "exclude_tools": (
                ["submit_feedback"]
                if os.getenv("CLC_DISABLE_SUBMIT_FEEDBACK", "").lower()
                in {"1", "true", "yes"}
                else []
            ),
        },
    },
)
