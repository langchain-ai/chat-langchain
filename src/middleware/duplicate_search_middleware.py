"""Suppress successful searches repeated within an agent invocation."""

import json
import operator
from typing import Annotated, Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command, Overwrite
from typing_extensions import NotRequired

SEARCH_TOOLS = {
    "langchainDocs_search_docs_by_lang_chain",
    "search_support_articles",
}


class DuplicateSearchState(AgentState):
    """Keep successful search keys separate from summarizable messages."""

    successful_searches: NotRequired[Annotated[dict[str, str], operator.or_]]


def _is_error_payload(content: Any) -> bool:
    """Recognize explicit errors in structured search output and text blocks."""
    if isinstance(content, str):
        try:
            return _is_error_payload(json.loads(content))
        except (ValueError, TypeError):
            return False
    if isinstance(content, list):
        return any(_is_error_payload(block) for block in content)
    if isinstance(content, dict):
        return (
            "error" in content
            or content.get("isError") is True
            or content.get("status") == "error"
            or (
                content.get("type") == "text" and _is_error_payload(content.get("text"))
            )
        )
    return False


class DuplicateSearchMiddleware(AgentMiddleware[DuplicateSearchState]):
    """Prevent identical searches after a successful result in this invocation."""

    state_schema = DuplicateSearchState

    def before_agent(
        self, state: DuplicateSearchState, runtime: Runtime
    ) -> dict[str, Any]:
        """Reset search tracking for each invocation, including checkpointed threads."""
        return {"successful_searches": Overwrite({})}

    def _duplicate(
        self, request: ToolCallRequest
    ) -> tuple[str | None, ToolMessage | None]:
        name = request.tool_call["name"]
        if name not in SEARCH_TOOLS:
            return None, None
        key = json.dumps([name, json.dumps(request.tool_call["args"], sort_keys=True)])
        previous_id = request.state.get("successful_searches", {}).get(key)
        if previous_id is None:
            return key, None
        return key, ToolMessage(
            content=(
                f"This identical search already succeeded (tool call {previous_id}). "
                "Use the earlier result or try a different query."
            ),
            name=name,
            tool_call_id=request.tool_call["id"],
        )

    def _record_success(
        self, request: ToolCallRequest, key: str | None, result: ToolMessage | Command
    ) -> ToolMessage | Command:
        if (
            key is not None
            and isinstance(result, ToolMessage)
            and result.status != "error"
            and not _is_error_payload(result.content)
            and not _is_error_payload(result.artifact)
        ):
            return Command(
                update={
                    "messages": [result],
                    "successful_searches": {key: request.tool_call["id"]},
                }
            )
        return result

    def wrap_tool_call(
        self, request: ToolCallRequest, handler
    ) -> ToolMessage | Command:
        """Execute a synchronous search only when it has not already succeeded."""
        key, duplicate = self._duplicate(request)
        if duplicate is not None:
            return duplicate
        return self._record_success(request, key, handler(request))

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler
    ) -> ToolMessage | Command:
        """Execute an asynchronous search only when it has not already succeeded."""
        key, duplicate = self._duplicate(request)
        if duplicate is not None:
            return duplicate
        return self._record_success(request, key, await handler(request))
