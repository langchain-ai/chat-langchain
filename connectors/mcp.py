"""Managed MCP connector declarations for Chat LangChain."""

from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolArg, StructuredTool
from managed_deepagents.connectors import (
    McpServersDefinition,
    define_mcp_servers,
)

MCP_DOCS_CHARACTER_BUDGET = 20_000
MCP_DOCS_TOOLS = frozenset(
    {
        "search_docs_by_lang_chain",
        "query_docs_filesystem_docs_by_lang_chain",
    }
)
_TRUNCATION_NOTE = (
    "[Result truncated; narrow the query or read a specific page for more detail.]"
)


def _truncate_text(text: str, budget: int) -> tuple[str, bool]:
    if len(text) <= budget:
        return text, False

    available = max(0, budget - len(_TRUNCATION_NOTE))
    boundary = text.rfind("\n", 0, available)
    if boundary <= 0:
        boundary = available
    return text[:boundary].rstrip() + _TRUNCATION_NOTE, True


def _truncate_content(content: Any) -> tuple[Any, bool]:
    if isinstance(content, str):
        return _truncate_text(content, MCP_DOCS_CHARACTER_BUDGET)

    if isinstance(content, list):
        remaining = MCP_DOCS_CHARACTER_BUDGET
        truncated = False
        bounded_content = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                bounded_text, block_truncated = _truncate_text(block.get("text", ""), remaining)
                bounded_content.append({**block, "text": bounded_text})
                remaining -= len(bounded_text)
                truncated = truncated or block_truncated
                if block_truncated:
                    break
            else:
                bounded_content.append(block)
        return bounded_content, truncated

    if isinstance(content, ToolMessage):
        bounded_content, truncated = _truncate_content(content.content)
        if truncated:
            return content.model_copy(update={"content": bounded_content}), True
    return content, False


def _wrap_docs_tool(tool: Any) -> Any:
    original_coroutine = tool.coroutine

    async def bounded_coroutine(
        runtime: Annotated[object | None, InjectedToolArg()] = None,
        **kwargs: Any,
    ) -> Any:
        result = await original_coroutine(runtime=runtime, **kwargs)
        if not isinstance(result, tuple) or len(result) != 2:
            return result
        content, artifact = result
        bounded_content, _ = _truncate_content(content)
        return bounded_content, artifact

    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        coroutine=bounded_coroutine,
        response_format=tool.response_format,
        metadata=tool.metadata,
        handle_tool_error=tool.handle_tool_error,
        return_direct=tool.return_direct,
    )


class _BoundedMcpServersDefinition(McpServersDefinition):
    def tools(self, ctx: dict[str, object] | None = None) -> list[Any]:
        tools = super().tools(ctx)
        return [
            _wrap_docs_tool(tool) if tool.name.rsplit("__", 1)[-1] in MCP_DOCS_TOOLS else tool
            for tool in tools
        ]


_base_connector = define_mcp_servers(
    prefix_tool_name_with_server_name=False,
    mcp_servers={
        "langchain-docs": {
            "transport": "http",
            "url": "https://docs.langchain.com/mcp",
        },
    },
)
connector = _BoundedMcpServersDefinition(_base_connector.config)
