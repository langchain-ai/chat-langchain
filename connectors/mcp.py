"""Managed MCP connector declarations for Chat LangChain."""

from managed_deepagents.connectors import define_mcp_servers

connector = define_mcp_servers(
    prefix_tool_name_with_server_name=False,
    mcp_servers={
        # TODO(docs-snapshot): the filesystem snapshot behind this server only
        # carries a curated subset of `integrations/**` — category `index.mdx`
        # files exist but most per-provider pages (`chat/`, `embeddings/`,
        # `document_loaders/file_loaders/` and `web_loaders/`, `vectorstores/`,
        # `retrievers/`, `tools/`) do not, so live docs pages the snapshot's own
        # index files link to cannot be read. Ask the snapshot generator owners
        # to sync the full published tree, and to return a machine-usable
        # fallback (`not_indexed: <path>; nearest indexed pages: <list>`)
        # instead of a bare shell error on a missed read.
        "langchain-docs": {
            "transport": "http",
            "url": "https://docs.langchain.com/mcp",
        },
    },
)
