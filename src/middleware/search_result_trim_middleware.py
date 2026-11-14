"""Trim documentation search chunks before they enter model context."""

import re
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command

SNIPPET_CHARS = 300
MAX_SECTION_CHARS = 1500
TRUNCATION_NOTICE = (
    "[snippets truncated - read the page with "
    "langchainDocs_query_docs_filesystem_docs_by_lang_chain before answering]"
)
READ_REMINDER_NAME = "docs_page_read_reminder"
STOPWORDS = {
    "python",
    "javascript",
    "langchain",
    "package",
    "how",
    "the",
    "and",
    "for",
    "with",
}
SEARCH_TOOL_SUFFIX = "search_docs_by_lang_chain"


def _is_docs_search(name: str) -> bool:
    """Match the docs search tool with or without its MCP server prefix."""
    return name == SEARCH_TOOL_SUFFIX or name.endswith("_" + SEARCH_TOOL_SUFFIX)


class SearchResultTrimMiddleware(AgentMiddleware):
    """Keep documentation search metadata and short content snippets."""

    def _trim_text(self, text: str, query: str = "") -> str:
        """Trim Content sections within Title-delimited search results."""
        sections = re.split(r"(?m)(?=^Title: )", text)
        for index, section in enumerate(sections):
            if not section.startswith("Title: "):
                continue
            sections[index] = re.sub(
                r"(^Content:[ \t]?)(.*?)(?=^(?:Link|Page|Content):|\Z)",
                lambda match: self._trim_content(match, query),
                section,
                flags=re.MULTILINE | re.DOTALL,
            )
        trimmed = "".join(sections)
        return trimmed + "\n" + TRUNCATION_NOTICE if trimmed != text else text

    def _trim_content(self, match: re.Match[str], query: str = "") -> str:
        """Limit a content body without consuming subsequent metadata lines."""
        header, body = match.groups()
        if len(body) <= SNIPPET_CHARS:
            return match.group(0)
        separator = "\n" if body.endswith("\n") else ""
        tokens = {
            token
            for token in re.findall(r"[^\W_]+(?:[-_][^\W_]+)*", query.lower())
            if len(token) >= 3 and token not in STOPWORDS
        }
        patterns = [
            re.compile(
                re.escape(token).replace("_", "[-_]").replace(r"\-", "[-_]"),
                re.IGNORECASE,
            )
            for token in tokens
        ]
        fragments = [body[:SNIPPET_CHARS].rstrip()]
        offset = 0
        for line in body.splitlines(keepends=True):
            remainder = line[max(0, SNIPPET_CHARS - offset) :]
            if any(pattern.search(remainder) for pattern in patterns):
                fragments.append(remainder.rstrip())
            offset += len(line)
        snippet = "...".join(fragments)
        return header + snippet[: MAX_SECTION_CHARS - 3] + "..." + separator

    def _trim_result(
        self, result: ToolMessage | Command, query: str = ""
    ) -> ToolMessage | Command:
        """Copy a tool message with only its text content trimmed."""
        if not isinstance(result, ToolMessage):
            return result
        content = result.content
        if isinstance(content, str):
            content = self._trim_text(content, query)
        else:
            content = [
                {**block, "text": self._trim_text(block["text"], query)}
                if isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
                else block
                for block in content
            ]
        return result.model_copy(update={"content": content})

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Trim synchronous docs search results and pass other tools through."""
        result = handler(request)
        if not _is_docs_search(request.tool_call["name"]):
            return result
        return self._trim_result(result, request.tool_call["args"].get("query", ""))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Trim asynchronous docs search results and pass other tools through."""
        result = await handler(request)
        if not _is_docs_search(request.tool_call["name"]):
            return result
        return self._trim_result(result, request.tool_call["args"].get("query", ""))

    @hook_config(can_jump_to=["model"])
    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Retry a final answer once when docs search was not followed by a read."""
        messages = state.get("messages", [])
        if (
            not messages
            or not isinstance(messages[-1], AIMessage)
            or messages[-1].tool_calls
        ):
            return None
        final_id = messages[-1].id
        if final_id is None:
            return None
        searched = False
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                break
            if (
                isinstance(message, SystemMessage)
                and message.name == READ_REMINDER_NAME
            ):
                return None
            if isinstance(message, ToolMessage):
                name = message.name or ""
                if message.status != "error" and (
                    name == "read_integration_page"
                    or name == "query_docs_filesystem_docs_by_lang_chain"
                    or name.endswith("_query_docs_filesystem_docs_by_lang_chain")
                ):
                    return None
                searched = searched or _is_docs_search(name)
        if not searched:
            return None
        return {
            "messages": [
                RemoveMessage(id=final_id),
                SystemMessage(
                    content="Read a relevant page with langchainDocs_query_docs_filesystem_docs_by_lang_chain or read_integration_page before answering. Search snippets cannot establish that a package, class, or feature does not exist.",
                    name=READ_REMINDER_NAME,
                ),
            ],
            "jump_to": "model",
        }

    @hook_config(can_jump_to=["model"])
    async def aafter_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Apply the same page-read reminder on the asynchronous final-answer path."""
        return self.after_model(state, runtime)
