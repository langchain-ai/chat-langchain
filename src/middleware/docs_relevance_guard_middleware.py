"""Mark documentation search results that miss every query term."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

_QUERY_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "also",
        "and",
        "are",
        "because",
        "before",
        "being",
        "between",
        "can",
        "could",
        "does",
        "for",
        "from",
        "how",
        "into",
        "more",
        "not",
        "should",
        "that",
        "the",
        "their",
        "there",
        "these",
        "this",
        "those",
        "through",
        "what",
        "when",
        "where",
        "which",
        "with",
        "would",
    }
)
_MARKER_TEMPLATE = (
    'NO MATCHING DOCUMENTATION: the docs search for "{query}" returned results '
    "that do not mention any term from the query. Do not answer from memory. Either "
    "re-query with different terms or tell the user the documentation does not cover this."
)


class DocsRelevanceGuardMiddleware(AgentMiddleware):
    """Mark documentation search results with no query-term matches."""

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Add a miss marker after an ungrounded documentation search."""
        if request.tool_call.get("name") != "search_docs_by_lang_chain":
            return await handler(request)

        result = await handler(request)
        if not isinstance(result, ToolMessage):
            return result

        query = self._query(request.tool_call.get("args", {}))
        significant_terms = self._significant_terms(query)
        if not significant_terms:
            return result

        result_text = self._content_text(result.content)
        if any(term in result_text.lower() for term in significant_terms):
            return result

        marker = _MARKER_TEMPLATE.format(query=query)
        return result.model_copy(update={"content": f"{marker}\n\n{result_text}"})

    def _query(self, args: Any) -> str:
        if isinstance(args, dict):
            query = args.get("query", "")
        else:
            query = ""
        return str(query).strip()

    def _significant_terms(self, query: str) -> set[str]:
        return {
            token
            for token in _QUERY_TOKEN_PATTERN.findall(query.lower())
            if len(token) > 2 and token not in _STOPWORDS
        }

    def _content_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        return str(content)


__all__ = ["DocsRelevanceGuardMiddleware"]
