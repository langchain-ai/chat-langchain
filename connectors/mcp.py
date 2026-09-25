"""Managed MCP connector declarations for Chat LangChain."""

import json
from typing import Any

from langchain_core.tools import StructuredTool
from managed_deepagents._connector import Connector
from managed_deepagents.connectors import define_mcp_servers

MAX_SEARCH_RESULT_BYTES = 24_000
SEARCH_TOOL_NAME = "search_docs_by_lang_chain"


def _serialized_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode())


def _search_entries(value: Any) -> list[dict[str, str]] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if isinstance(value, dict):
        value = value.get("results")
    if not isinstance(value, list):
        return None

    entries = []
    for item in value:
        if not isinstance(item, dict):
            return None
        title = item.get("title")
        link = item.get("link") or item.get("url")
        if not isinstance(title, str) or not isinstance(link, str):
            return None
        entries.append({"title": title, "link": link})
    return entries


def constrain_search_result(
    value: Any, max_bytes: int = MAX_SEARCH_RESULT_BYTES
) -> Any:
    """Limit oversized search results while retaining complete title/link entries."""
    if _serialized_size(value) <= max_bytes:
        return value

    entries = _search_entries(value)
    if entries is None:
        return value

    selected: list[dict[str, str]] = []
    for entry in entries:
        omitted = len(entries) - len(selected) - 1
        candidate = selected + [entry]
        output = json.dumps(candidate, ensure_ascii=False, separators=(",", ":"))
        output += f"\nOmitted {omitted} entries."
        if len(output.encode()) > max_bytes:
            break
        selected.append(entry)

    omitted = len(entries) - len(selected)
    output = json.dumps(selected, ensure_ascii=False, separators=(",", ":"))
    output += f"\nOmitted {omitted} entries."
    while len(output.encode()) > max_bytes and selected:
        selected.pop()
        omitted = len(entries) - len(selected)
        output = json.dumps(selected, ensure_ascii=False, separators=(",", ":"))
        output += f"\nOmitted {omitted} entries."
    return output


def _bounded_tool(tool: Any) -> Any:
    async def invoke(**kwargs: Any) -> Any:
        result = await tool.ainvoke(kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            return constrain_search_result(result[0]), result[1]
        return constrain_search_result(result)

    return StructuredTool.from_function(
        coroutine=invoke,
        name=tool.name,
        description=tool.description or tool.name,
        args_schema=tool.args_schema,
        infer_schema=False,
        return_direct=tool.return_direct,
        response_format=tool.response_format,
    )


class BoundedMcpConnector(Connector):
    """Apply the search result bound before MCP tools enter the agent."""

    kind = "mcp_servers"

    def __init__(self, definition: Any):
        self.definition = definition

    def tools(self, ctx: dict[str, object] | None = None) -> list[Any]:
        del ctx
        tools = self.definition.tools()
        return [
            _bounded_tool(tool) if tool.name == SEARCH_TOOL_NAME else tool
            for tool in tools
        ]


_mcp_connector = define_mcp_servers(
    prefix_tool_name_with_server_name=False,
    mcp_servers={
        "langchain-docs": {
            "transport": "http",
            "url": "https://docs.langchain.com/mcp",
        },
    },
)

connector = BoundedMcpConnector(_mcp_connector)

__all__ = ["MAX_SEARCH_RESULT_BYTES", "constrain_search_result", "connector"]
