"""Keep documentation search results concise without changing page reads."""

import re
from collections.abc import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

_HIT_HEADER = re.compile(
    r"^(Title:[^\n]*\nLink:[^\n]*\nPage:[^\n]*\nContent:)[ \t]*",
    re.MULTILINE,
)
_MAX_HITS = 6
_MAX_CONTENT_CHARS = 200


def _trim_text(text: str, remaining_hits: int) -> tuple[str, int]:
    matches = list(_HIT_HEADER.finditer(text))
    if not matches:
        return text, 0
    if not remaining_hits:
        return "", 0

    hits = []
    for index, match in enumerate(matches[:remaining_hits]):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        content = text[match.end() : end].strip()
        snippet = content[:_MAX_CONTENT_CHARS]
        if len(content) > _MAX_CONTENT_CHARS:
            snippet += "..."
        hits.append(f"{match.group(1)} {snippet}")
    return text[: matches[0].start()] + "\n\n".join(hits), len(hits)


class SearchResultTrimMiddleware(AgentMiddleware):
    """Limit docs search hits and snippets while preserving other tool results."""

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Trim documentation search output after the asynchronous handler runs."""
        result = await handler(request)
        return self._trim_result(request, result)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Trim documentation search output after the synchronous handler runs."""
        result = handler(request)
        return self._trim_result(request, result)

    def _trim_result(
        self, request: ToolCallRequest, result: ToolMessage | Command
    ) -> ToolMessage | Command:
        if request.tool_call["name"] != "search_docs_by_lang_chain" or not isinstance(
            result, ToolMessage
        ):
            return result

        content: str | list[str | dict]
        if isinstance(result.content, str):
            content, _ = _trim_text(result.content, _MAX_HITS)
        else:
            content = []
            remaining_hits = _MAX_HITS
            for block in result.content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "text"
                    and isinstance(block.get("text"), str)
                ):
                    text, hit_count = _trim_text(block["text"], remaining_hits)
                    remaining_hits -= hit_count
                    if text or not block["text"]:
                        content.append({**block, "text": text})
                else:
                    content.append(block)
        return result.model_copy(update={"content": content})
