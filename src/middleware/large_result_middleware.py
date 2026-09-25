"""Recover oversized tool results spilled to the runtime filesystem."""

import re
from collections.abc import Iterable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

LARGE_RESULT_PATH_PATTERN = re.compile(r"/large_tool_results/[^\s`\"'<>]+")


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return " ".join(_content_text(value) for value in content.values())
    if isinstance(content, Iterable) and not isinstance(content, (bytes, bytearray)):
        return " ".join(_content_text(value) for value in content)
    return str(content)


def extract_large_result_path(content: Any) -> str | None:
    """Return the first spilled-result path in tool content."""
    match = LARGE_RESULT_PATH_PATTERN.search(_content_text(content))
    return match.group(0).rstrip(".,;:)") if match else None


class LargeResultMiddleware(AgentMiddleware[AgentState]):
    """Tell the model to read tool results spilled by the runtime."""

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler,
    ) -> ToolMessage | Command:
        """Append a read instruction when the runtime spills tool output."""
        result = await handler(request)
        if not isinstance(result, ToolMessage):
            return result

        path = extract_large_result_path(result.content)
        if path is None:
            return result

        directive = (
            f"The tool output was too large and is stored at {path}. "
            "Call read_file on exactly this path next with offset=0 and limit=200; "
            "page further only as needed."
        )
        if isinstance(result.content, str):
            content = f"{result.content}\n\n{directive}"
        elif isinstance(result.content, list):
            content = [*result.content, {"type": "text", "text": directive}]
        else:
            content = f"{_content_text(result.content)}\n\n{directive}"
        return result.model_copy(update={"content": content})


__all__ = ["LargeResultMiddleware", "extract_large_result_path"]
