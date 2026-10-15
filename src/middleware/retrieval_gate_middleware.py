"""Retrieval gate: block substantive answers that skipped docs retrieval.

The docs_agent system prompt forbids answering technical questions from memory,
but the model is free to ignore it and emit fabricated code snippets or
``docs.langchain.com`` URLs without ever calling a retrieval tool. This
middleware enforces the rule structurally: after the model produces an
assistant message, if that message looks like a substantive technical answer
yet no retrieval tool was called this turn, the model is re-invoked once with a
reminder to search first (or ask a clarifying question / refuse).
"""

from __future__ import annotations

import dataclasses
import logging
import re
from typing import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, HumanMessage

logger = logging.getLogger(__name__)

#: Tool-name prefixes that count as a docs/support retrieval call.
RETRIEVAL_TOOL_PREFIXES = (
    "search_docs_",
    "search_support_articles",
    "query_docs_filesystem_",
    "query_filesystem_",
)

#: Prose length above which an answer is treated as substantive.
SUBSTANTIVE_PROSE_CHARS = 300

_CODE_FENCE = re.compile(r"```")
_RELEVANT_DOCS = "**Relevant docs:**"
_DOCS_URL = "docs.langchain.com"

_RETRY_REMINDER = (
    "You did not call any docs retrieval tool this turn. The system prompt "
    "forbids answering from memory. Either call `search_docs_by_lang_chain` or "
    "`query_docs_filesystem_docs_by_lang_chain` now, or respond with a "
    "clarifying question / scope refusal."
)


class RetrievalGateMiddleware(AgentMiddleware):
    """Require at least one retrieval call before emitting a substantive answer."""

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Re-invoke the model once when it answers substantively without retrieving."""
        response = await handler(request)

        if self._turn_retrieval_count(request.messages) > 0:
            return response
        if not self._is_substantive_answer(response):
            return response

        logger.warning(
            "Retrieval gate: substantive answer with zero retrieval calls this "
            "turn; re-invoking model with search reminder."
        )
        retry_request = dataclasses.replace(
            request,
            messages=[*request.messages, HumanMessage(content=_RETRY_REMINDER)],
        )
        retry_response = await handler(retry_request)

        if self._turn_retrieval_count(retry_request.messages) == 0 and (
            self._is_substantive_answer(retry_response)
        ):
            logger.warning(
                "Retrieval gate: second attempt still produced a substantive "
                "answer with zero retrieval calls; returning as-is."
            )
        return retry_response

    def _turn_retrieval_count(self, messages: list) -> int:
        """Count retrieval tool calls since the most recent human message."""
        count = 0
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                break
            for tool_call in getattr(message, "tool_calls", None) or []:
                name = tool_call.get("name", "") if isinstance(tool_call, dict) else ""
                if name.startswith(RETRIEVAL_TOOL_PREFIXES):
                    count += 1
        return count

    def _is_substantive_answer(self, response: ModelResponse) -> bool:
        """Return True when the model's final message reads like a technical answer."""
        message = self._final_ai_message(response)
        if message is None or getattr(message, "tool_calls", None):
            return False
        text = self._message_text(message)
        if _CODE_FENCE.search(text):
            return True
        if _RELEVANT_DOCS in text:
            return True
        if _DOCS_URL in text:
            return True
        return len(text) > SUBSTANTIVE_PROSE_CHARS

    def _final_ai_message(self, response: ModelResponse) -> AIMessage | None:
        """Return the last AIMessage from the model response, if any."""
        for message in reversed(getattr(response, "result", []) or []):
            if isinstance(message, AIMessage):
                return message
        return None

    def _message_text(self, message: AIMessage) -> str:
        """Flatten an assistant message's content to plain text."""
        content = message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
            return "".join(parts)
        return ""


__all__ = ["RetrievalGateMiddleware", "RETRIEVAL_TOOL_PREFIXES"]
