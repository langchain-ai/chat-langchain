"""Cap oversized discovery tool payloads before they enter the message history."""
import logging
from typing import Any, Iterable

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

logger = logging.getLogger(__name__)

#: Discovery (search) tools only. Read tools must stay uncapped so answers stay grounded.
CAPPED_TOOLS = frozenset(
    {
        "search_docs_by_lang_chain",
        "search_support_articles",
    }
)

MAX_RESULTS = 10
MAX_ENTRY_CHARS = 1_500
MAX_TOTAL_CHARS = 30_000

ENTRY_SEPARATOR = "\n\n"


class ToolResultCapMiddleware(AgentMiddleware[AgentState]):
    """Truncate search tool results so a single payload cannot fill the context window."""

    def __init__(
        self,
        tool_names: Iterable[str] = CAPPED_TOOLS,
        max_results: int = MAX_RESULTS,
        max_entry_chars: int = MAX_ENTRY_CHARS,
        max_total_chars: int = MAX_TOTAL_CHARS,
    ):
        super().__init__()
        self.tool_names = frozenset(tool_names)
        self.max_results = max_results
        self.max_entry_chars = max_entry_chars
        self.max_total_chars = max_total_chars

    def _truncate_entry(self, entry: str) -> str:
        if len(entry) <= self.max_entry_chars:
            return entry
        return f"{entry[: self.max_entry_chars].rstrip()} [truncated: entry shortened]"

    def _cap_total(self, text: str) -> str:
        if len(text) <= self.max_total_chars:
            return text
        return (
            f"{text[: self.max_total_chars].rstrip()}"
            f"{ENTRY_SEPARATOR}[truncated: result exceeded size cap — refine your query]"
        )

    def _cap_text(self, text: str) -> str:
        entries = [entry for entry in text.split(ENTRY_SEPARATOR) if entry.strip()]

        # A single blob (e.g. the JSON support-article payload) has no result
        # boundaries to slice on, so only the overall ceiling applies.
        if len(entries) <= 1:
            return self._cap_total(text)

        kept = entries[: self.max_results]
        omitted = len(entries) - len(kept)
        capped = ENTRY_SEPARATOR.join(self._truncate_entry(entry) for entry in kept)
        if omitted:
            capped = (
                f"{capped}{ENTRY_SEPARATOR}"
                f"[truncated: {omitted} more results omitted — refine your query]"
            )
        return self._cap_total(capped)

    def _cap_content(self, content: Any) -> Any:
        if isinstance(content, str):
            return self._cap_text(content)

        if isinstance(content, list):
            return [
                {**block, "text": self._cap_text(block["text"])}
                if isinstance(block, dict) and isinstance(block.get("text"), str)
                else block
                for block in content
            ]

        return content

    def _cap_message(self, message: ToolMessage, tool_name: str) -> ToolMessage:
        original = message.content
        capped = self._cap_content(original)
        if capped == original:
            return message

        logger.info(
            "Capped %s result from %s to %s chars",
            tool_name,
            len(str(original)),
            len(str(capped)),
        )
        return message.model_copy(update={"content": capped})

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler,
    ) -> ToolMessage | Command:
        result = await handler(request)
        tool_name = request.tool_call.get("name", "")
        if tool_name not in self.tool_names:
            return result

        if isinstance(result, ToolMessage):
            return self._cap_message(result, tool_name)

        if isinstance(result, Command) and isinstance(result.update, dict):
            messages = result.update.get("messages")
            if isinstance(messages, list):
                result.update["messages"] = [
                    self._cap_message(message, tool_name)
                    if isinstance(message, ToolMessage)
                    else message
                    for message in messages
                ]

        return result


__all__ = ["ToolResultCapMiddleware"]
