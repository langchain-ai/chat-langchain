"""Require content retrieval before lengthy technical answers."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)

DISCOVERY_TO_CONTENT = {
    "search_docs_by_lang_chain": "query_docs_filesystem_docs_by_lang_chain",
    "search_support_articles": "get_support_article_content",
}


class GroundingGateState(AgentState, total=False):
    grounding_discovery_tools: list[str]
    grounding_content_tools: list[str]
    grounding_discovered_references: list[str]
    grounding_gate_triggered: bool


class GroundingGateMiddleware(AgentMiddleware[AgentState]):
    """Re-prompt lengthy search-only answers to retrieve source content."""

    state_schema = GroundingGateState

    def before_agent(
        self, state: GroundingGateState, runtime: Runtime
    ) -> dict[str, Any]:
        """Reset grounding records for each agent run."""
        return {
            "grounding_discovery_tools": [],
            "grounding_content_tools": [],
            "grounding_discovered_references": [],
            "grounding_gate_triggered": False,
        }

    def _record_tool(self, state: dict[str, Any], tool_name: str) -> None:
        if tool_name in DISCOVERY_TO_CONTENT:
            state.setdefault("grounding_discovery_tools", []).append(tool_name)
        elif tool_name in DISCOVERY_TO_CONTENT.values():
            state.setdefault("grounding_content_tools", []).append(tool_name)

    def _references_from_result(self, result: Any) -> list[str]:
        content = result.content if isinstance(result, ToolMessage) else str(result)
        references: list[str] = []
        try:
            payload = json.loads(content) if isinstance(content, str) else content
        except (TypeError, ValueError):
            payload = None

        def collect(value: Any, key: str | None = None) -> None:
            if isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    if nested_key in {"path", "page", "url", "id", "identifier"}:
                        collect(nested_value, nested_key)
                    elif isinstance(nested_value, (dict, list)):
                        collect(nested_value)
            elif isinstance(value, list):
                for item in value:
                    collect(item)
            elif isinstance(value, str) and key in {"path", "page", "url", "id", "identifier"}:
                references.append(value)

        collect(payload)
        if not references and isinstance(content, str):
            references.extend(re.findall(r"[^\s\"']+\.mdx|[0-9a-f]{8}-[0-9a-f-]{27,36}", content))
        return list(dict.fromkeys(references))[:8]

    def _record_result(self, state: dict[str, Any], tool_name: str, result: Any) -> None:
        if tool_name in DISCOVERY_TO_CONTENT:
            state.setdefault("grounding_discovered_references", []).extend(
                self._references_from_result(result)
            )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage],
    ) -> ToolMessage:
        """Track a tool call and its discovery references."""
        tool_name = request.tool_call.get("name", "unknown_tool")
        self._record_tool(request.state, tool_name)
        result = handler(request)
        self._record_result(request.state, tool_name, result)
        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage]],
    ) -> ToolMessage:
        """Track an async tool call and its discovery references."""
        tool_name = request.tool_call.get("name", "unknown_tool")
        self._record_tool(request.state, tool_name)
        result = await handler(request)
        self._record_result(request.state, tool_name, result)
        return result

    def _answer_text(self, message: AIMessage) -> str:
        if isinstance(message.content, str):
            return message.content
        return message.text

    def _correction_message(self, state: dict[str, Any]) -> HumanMessage:
        discoveries = list(dict.fromkeys(state.get("grounding_discovered_references", [])))
        references = ", ".join(discoveries) or "the most relevant result from the discovery tools"
        missing_tools = [
            content_tool
            for discovery_tool, content_tool in DISCOVERY_TO_CONTENT.items()
            if discovery_tool in state.get("grounding_discovery_tools", [])
            and content_tool not in state.get("grounding_content_tools", [])
        ]
        tools = " and ".join(f"`{tool}`" for tool in missing_tools)
        return HumanMessage(
            content=(
                "Before finalizing this technical answer, retrieve the full source content. "
                f"Use {tools} with the most relevant discovered reference ({references}), "
                "then rewrite the answer using only the retrieved content."
            )
        )

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Re-prompt a substantive answer that lacks source content."""
        messages = state.get("messages", [])
        if not messages or state.get("grounding_gate_triggered", False):
            return None
        last_message = messages[-1]
        if not isinstance(last_message, AIMessage) or last_message.tool_calls:
            return None
        if len(self._answer_text(last_message)) <= 600:
            return None

        discovery_tools = set(state.get("grounding_discovery_tools", []))
        content_tools = set(state.get("grounding_content_tools", []))
        missing_content = {
            content_tool
            for discovery_tool, content_tool in DISCOVERY_TO_CONTENT.items()
            if discovery_tool in discovery_tools and content_tool not in content_tools
        }
        if not missing_content:
            return None

        logger.info("Grounding gate triggered for search-only technical answer")
        return {
            "messages": [self._correction_message(state)],
            "grounding_gate_triggered": True,
        }


__all__ = ["GroundingGateMiddleware"]
