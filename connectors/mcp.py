"""Managed MCP connector declarations for Chat LangChain."""

import json
import os
from collections.abc import Mapping
from typing import Any

from langchain_core.tools import StructuredTool
from managed_deepagents.connectors import McpServersDefinition, define_mcp_servers

DOCS_SEARCH_TOOL_NAME = "search_docs_by_lang_chain"
DOCS_SEARCH_MAX_HITS = int(os.getenv("DOCS_SEARCH_MAX_HITS", "8"))
DOCS_SEARCH_MAX_CHARS = int(os.getenv("DOCS_SEARCH_MAX_CHARS", "12000"))
DOCS_SEARCH_EXCERPT_CHARS = 1200


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(part for item in value if (part := _text(item)))
    if isinstance(value, Mapping):
        if isinstance(value.get("text"), str):
            return value["text"]
        return " ".join(part for item in value.values() if (part := _text(item)))
    return str(value) if value is not None else ""


def _search_hits(result: Any) -> list[Mapping[str, Any]]:
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            return []
    if isinstance(result, Mapping):
        for key in ("results", "hits", "data"):
            if isinstance(result.get(key), list):
                result = result[key]
                break
    return [hit for hit in result if isinstance(hit, Mapping)] if isinstance(result, list) else []


def bound_docs_search_result(result: Any, max_chars: int = DOCS_SEARCH_MAX_CHARS) -> str:
    """Keep ranked docs search output within the middleware result budget."""
    bounded: list[dict[str, str]] = []
    for hit in _search_hits(result)[:DOCS_SEARCH_MAX_HITS]:
        bounded.append(
            {
                "Title": _text(hit.get("title") or hit.get("name")),
                "Link": _text(hit.get("link") or hit.get("url")),
                "Page": _text(hit.get("page") or hit.get("path") or hit.get("source")),
                "Content excerpt": _text(
                    hit.get("content") or hit.get("snippet") or hit.get("description")
                )[:DOCS_SEARCH_EXCERPT_CHARS],
            }
        )
    note = "Full page content must be read with query_docs_filesystem_docs_by_lang_chain."
    while bounded and len(json.dumps({"results": bounded, "note": note})) > max_chars:
        longest = max(range(len(bounded)), key=lambda index: len(bounded[index]["Content excerpt"]))
        if not bounded[longest]["Content excerpt"]:
            bounded.pop()
        else:
            bounded[longest]["Content excerpt"] = bounded[longest]["Content excerpt"][:-200]
    return json.dumps({"results": bounded, "note": note}, ensure_ascii=False)


def _bounded_tool(tool: Any, max_chars: int = DOCS_SEARCH_MAX_CHARS) -> Any:
    async def invoke(query: str) -> str:
        return bound_docs_search_result(await tool.ainvoke({"query": query}), max_chars=max_chars)

    return StructuredTool.from_function(
        coroutine=invoke,
        name=DOCS_SEARCH_TOOL_NAME,
        description=tool.description,
        args_schema=tool.args_schema,
    )


class BoundedDocsMcpDefinition(McpServersDefinition):
    """MCP definition that bounds the documentation search result."""

    def tools(self, ctx: Mapping[str, object] | None = None) -> list[object]:
        """Load MCP tools and bound the documentation search tool."""
        return [
            _bounded_tool(tool) if getattr(tool, "name", None) == DOCS_SEARCH_TOOL_NAME else tool
            for tool in super().tools(ctx)
        ]

_mcp_definition = define_mcp_servers(
    prefix_tool_name_with_server_name=False,
    mcp_servers={
        "langchain-docs": {
            "transport": "http",
            "url": "https://docs.langchain.com/mcp",
        },
    },
)

connector = BoundedDocsMcpDefinition(_mcp_definition.config)
