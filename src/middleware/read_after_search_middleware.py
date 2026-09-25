"""Enforce reading search results before final documentation answers."""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import (
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

_SEARCH_PAIRS = {
    "search_docs_by_lang_chain": (
        "query_docs_filesystem_docs_by_lang_chain",
        "documentation paths",
    ),
    "search_support_articles": ("get_support_article_content", "article IDs"),
}


class ReadAfterSearchMiddleware(AgentMiddleware):
    """Nudge the model once when it tries to answer before reading search results."""

    def __init__(self) -> None:
        """Initialize run-local search tracking."""
        self._outstanding: dict[str, dict[str, list[str]]] = {}
        self._nudged: dict[str, set[str]] = {}

    def _run_key(self, runtime: Runtime) -> str:
        execution_info = runtime.execution_info
        if execution_info and execution_info.run_id:
            return execution_info.run_id
        if execution_info and execution_info.thread_id:
            return execution_info.thread_id
        return str(id(runtime))

    def _tool_name(self, request: ToolCallRequest) -> str:
        return request.tool_call.get("name", "")

    def _tool_content(self, result: ToolMessage | Command[Any]) -> str:
        content = getattr(result, "content", "")
        return content if isinstance(content, str) else str(content)

    def _candidates(self, tool_name: str, content: str) -> list[str]:
        if tool_name == "search_support_articles":
            try:
                payload = json.loads(content)
                return [
                    str(article["id"])
                    for article in payload.get("articles", [])
                    if article.get("id")
                ]
            except (json.JSONDecodeError, TypeError, AttributeError):
                return re.findall(r'"id"\s*:\s*"([^"]+)"', content)

        paths = re.findall(r"(?:/|\.\.?/)[^\s'\"`<>]+\.(?:mdx?|rst|html?)", content)
        return list(dict.fromkeys(paths))

    def _record_tool(
        self, request: ToolCallRequest, result: ToolMessage | Command[Any]
    ) -> None:
        tool_name = self._tool_name(request)
        run_key = self._run_key(request.runtime)
        if tool_name in _SEARCH_PAIRS:
            self._outstanding.setdefault(run_key, {})[tool_name] = self._candidates(
                tool_name, self._tool_content(result)
            )
        else:
            for search_name, (read_name, _) in _SEARCH_PAIRS.items():
                if tool_name == read_name:
                    self._outstanding.setdefault(run_key, {}).pop(search_name, None)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Track searches and clear them when their paired read runs."""
        result = handler(request)
        self._record_tool(request, result)
        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Track searches and clear them when their paired read runs asynchronously."""
        result = await handler(request)
        self._record_tool(request, result)
        return result

    def _model_messages(self, response: ModelResponse[Any]) -> list[AnyMessage]:
        return list(response.result)

    def _is_final_response(self, response: ModelResponse[Any]) -> bool:
        messages = self._model_messages(response)
        return bool(
            messages
            and isinstance(messages[-1], AIMessage)
            and not messages[-1].tool_calls
        )

    def _nudge_instruction(self, search_name: str, candidates: list[str]) -> str:
        read_name, label = _SEARCH_PAIRS[search_name]
        candidate_text = (
            ", ".join(candidates) or "the candidates returned by the search"
        )
        return (
            f"Before answering, you must call {read_name} for the {label} from "
            f"{search_name}: {candidate_text}. Search results alone are not grounding."
        )

    def _response_with_nudges(
        self,
        request: ModelRequest,
        response: ModelResponse[Any],
        handler: Callable[[ModelRequest], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        run_key = self._run_key(request.runtime)
        outstanding = self._outstanding.get(run_key, {})
        nudged = self._nudged.setdefault(run_key, set())
        if not self._is_final_response(response):
            return response

        for search_name, candidates in list(outstanding.items()):
            if search_name in nudged:
                continue
            nudged.add(search_name)
            messages = [*request.messages, *self._model_messages(response)]
            messages.append(
                HumanMessage(content=self._nudge_instruction(search_name, candidates))
            )
            response = handler(request.override(messages=messages))
        return response

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        """Require paired reads before returning a final model response."""
        return self._response_with_nudges(request, handler(request), handler)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        """Require paired reads before returning a final model response asynchronously."""
        response = await handler(request)
        run_key = self._run_key(request.runtime)
        outstanding = self._outstanding.get(run_key, {})
        nudged = self._nudged.setdefault(run_key, set())
        if not self._is_final_response(response):
            return response

        for search_name, candidates in list(outstanding.items()):
            if search_name in nudged:
                continue
            nudged.add(search_name)
            messages = [*request.messages, *self._model_messages(response)]
            messages.append(
                HumanMessage(content=self._nudge_instruction(search_name, candidates))
            )
            response = await handler(request.override(messages=messages))
        return response

    def after_agent(self, state: dict[str, Any], runtime: Runtime) -> None:
        """Release run-local search state after the graph finishes."""
        run_key = self._run_key(runtime)
        self._outstanding.pop(run_key, None)
        self._nudged.pop(run_key, None)

    async def aafter_agent(self, state: dict[str, Any], runtime: Runtime) -> None:
        """Release run-local search state after the graph finishes asynchronously."""
        self.after_agent(state, runtime)


__all__ = ["ReadAfterSearchMiddleware"]
