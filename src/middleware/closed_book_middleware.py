"""Withhold what the retrieval tools return, for probes of what a model answers from memory.

With the closed-book middleware in the stack, every call to a tool in CLOSED_BOOK_TOOLS (the docs
search and filesystem tools, the support-article tools, pricing and link checks) gets
CLOSED_BOOK_RESULT instead of running, so an answer can only come from what the model already
knows. The agent's tool list is unchanged. agent.py adds it only when POLICY_CLOSED_BOOK is set.
"""

from __future__ import annotations

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

CLOSED_BOOK_ENV = "POLICY_CLOSED_BOOK"
CLOSED_BOOK_TOOLS = (
    "search_docs_by_lang_chain",
    "query_docs_filesystem_docs_by_lang_chain",
    "search_support_articles",
    "get_support_article_content",
    "fetch_langchain_pricing",
    "check_links",
)
CLOSED_BOOK_RESULT = "No results."


def is_closed(tool_name: str) -> bool:
    """Whether a tool is withheld, by name or by an MCP server's prefixed name."""
    return any(tool_name == name or tool_name.endswith(f"_{name}") for name in CLOSED_BOOK_TOOLS)


class ClosedBookMiddleware(AgentMiddleware[AgentState]):
    """Answers each retrieval tool call with CLOSED_BOOK_RESULT; other calls run as usual."""

    def _withheld(self, request: ToolCallRequest) -> ToolMessage | None:
        call = request.tool_call
        if not is_closed(call.get("name", "")):
            return None
        return ToolMessage(
            content=CLOSED_BOOK_RESULT, tool_call_id=call.get("id", ""), name=call.get("name")
        )

    def wrap_tool_call(self, request: ToolCallRequest, handler) -> ToolMessage | Command:
        """Withhold a retrieval call's result; run any other call."""
        return self._withheld(request) or handler(request)

    async def awrap_tool_call(self, request: ToolCallRequest, handler) -> ToolMessage | Command:
        """Withhold a retrieval call's result; run any other call."""
        return self._withheld(request) or await handler(request)
