"""Keep documentation tool results scoped to the user's language."""

import os
import re
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from src.utils.doc_language import DocLanguage, infer_doc_language

SEARCH_TOOL = "search_docs_by_lang_chain"
FILESYSTEM_TOOL = "query_docs_filesystem_docs_by_lang_chain"
_BLOCK_START = re.compile(r"(?m)^(?=(?:Title|Page):\s)")
_LANGUAGE_PATHS = {
    "python": "/oss/python/",
    "javascript": "/oss/javascript/",
}


class DocsLanguageFilterMiddleware(AgentMiddleware[AgentState]):
    """Filter documentation results and reads to the inferred language."""

    def _enabled(self) -> bool:
        return os.getenv("DOCS_LANGUAGE_FILTER_ENABLED", "true").lower() not in {
            "0",
            "false",
            "no",
            "off",
        }

    def _tool_name(self, request: ToolCallRequest) -> str:
        return request.tool_call.get("name", "")

    def _tool_message(self, request: ToolCallRequest, content: str) -> ToolMessage:
        return ToolMessage(
            content=content,
            name=self._tool_name(request),
            tool_call_id=request.tool_call.get("id", ""),
        )

    def _filesystem_path_is_wrong(
        self, request: ToolCallRequest, language: DocLanguage
    ) -> bool:
        args: Any = request.tool_call.get("args", {})
        command = args.get("command", "") if isinstance(args, dict) else ""
        if not isinstance(command, str):
            return False
        wrong_path = _LANGUAGE_PATHS["javascript" if language == "python" else "python"]
        return wrong_path in command

    def _filter_search_result(self, content: Any, language: DocLanguage) -> Any:
        if not isinstance(content, str):
            return content

        starts = [match.start() for match in _BLOCK_START.finditer(content)]
        if not starts:
            return content
        blocks = [
            content[start:end]
            for start, end in zip(starts, starts[1:] + [len(content)])
        ]
        prefix = content[: starts[0]]
        wrong_path = _LANGUAGE_PATHS["javascript" if language == "python" else "python"]
        wrong_blocks = [
            block
            for block in blocks
            if ("Page:" in block or "Link:" in block) and wrong_path in block
        ]
        if not wrong_blocks:
            return content
        relevant_blocks = [
            block for block in blocks if "Page:" in block or "Link:" in block
        ]
        if relevant_blocks and len(wrong_blocks) == len(relevant_blocks):
            return content
        kept_blocks = [block for block in blocks if block not in wrong_blocks]
        filtered = prefix + "".join(kept_blocks)
        label = "JavaScript" if language == "python" else "Python"
        return (
            f"{filtered.rstrip()}\n\n"
            f"[{label} documentation filter: withheld {len(wrong_blocks)} "
            "wrong-language page(s).]"
        )

    def _filter_result(
        self, request: ToolCallRequest, result: ToolMessage, language: DocLanguage
    ) -> ToolMessage:
        if self._tool_name(request) != SEARCH_TOOL:
            return result
        content = self._filter_search_result(result.content, language)
        if content == result.content:
            return result
        return result.model_copy(update={"content": content})

    def _preflight(self, request: ToolCallRequest) -> ToolMessage | None:
        if not self._enabled():
            return None
        language = infer_doc_language(request.state.get("messages", []))
        if language is None:
            return None
        if self._tool_name(
            request
        ) != FILESYSTEM_TOOL or not self._filesystem_path_is_wrong(request, language):
            return None
        label = "JavaScript" if language == "python" else "Python"
        return self._tool_message(
            request,
            f"Withheld filesystem read: the requested path is a {label} documentation page, "
            f"but this turn is scoped to {language} documentation.",
        )

    def _handle(self, request: ToolCallRequest, result: ToolMessage):
        if not self._enabled():
            return result
        language = infer_doc_language(request.state.get("messages", []))
        if language is None:
            return result
        if self._tool_name(request) == FILESYSTEM_TOOL:
            return result
        if self._tool_name(request) == SEARCH_TOOL:
            return self._filter_result(request, result, language)
        return result

    def wrap_tool_call(self, request, handler):
        """Filter synchronous documentation tool results."""
        if blocked := self._preflight(request):
            return blocked
        result = handler(request)
        return (
            self._handle(request, result) if isinstance(result, ToolMessage) else result
        )

    async def awrap_tool_call(self, request, handler) -> ToolMessage | Command:
        """Filter asynchronous documentation tool results."""
        if blocked := self._preflight(request):
            return blocked
        result = await handler(request)
        return (
            self._handle(request, result) if isinstance(result, ToolMessage) else result
        )


__all__ = ["DocsLanguageFilterMiddleware"]
