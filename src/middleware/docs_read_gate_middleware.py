"""Read-before-answer gate for the docs agent.

``search_docs_by_lang_chain`` returns titles and links only (see
``instructions.md`` line 36). This middleware makes the mandated follow-up read
with ``query_docs_filesystem_docs_by_lang_chain`` a hard requirement instead of
a prose instruction the model can skip.
"""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain_core.messages import SystemMessage
from langgraph.runtime import Runtime

LOCATOR_TOOL_PREFIXES = ("search_docs_by_lang_chain",)
CONTENT_READ_TOOL_PREFIXES = ("query_docs_filesystem",)
CONTENT_READ_TOOL_NAMES = ("grep",)

#: Answers at or above this length, or containing a fenced code block, are
#: treated as substantive and must be grounded in real page content.
SUBSTANTIVE_CHARS = 400
CODE_FENCE = "```"

READ_NUDGE = (
    "You have not read any documentation page content this turn. "
    "`search_docs_by_lang_chain` returns titles and links only and is never a "
    "source for answer content. Call "
    "`query_docs_filesystem_docs_by_lang_chain` on the most relevant search "
    "result and ground your answer - especially every code example, API name, "
    "and package name - in what that page actually says. If no retrieved page "
    "covers the question, say the docs do not cover it instead of writing code "
    "from memory."
)


class DocsReadGateMiddleware(AgentMiddleware):
    """Block substantive answers that rest on locator search results alone."""

    @hook_config(can_jump_to=["model"])
    def after_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Send the model back to read a docs page when it answers from titles."""
        turn = self._messages_since_last_human(state.get("messages", []))
        if not turn:
            return None

        last = turn[-1]
        if getattr(last, "type", None) != "ai" or getattr(last, "tool_calls", None):
            return None

        if self._nudge_already_sent(turn):
            return None

        names = self._tool_names(turn)
        used_locator = any(n.startswith(LOCATOR_TOOL_PREFIXES) for n in names)
        read_content = any(
            n.startswith(CONTENT_READ_TOOL_PREFIXES) or n in CONTENT_READ_TOOL_NAMES
            for n in names
        )
        if not used_locator or read_content:
            return None

        if not self._is_substantive(getattr(last, "content", None)):
            return None

        return {
            "jump_to": "model",
            "messages": [SystemMessage(content=READ_NUDGE)],
        }

    def _messages_since_last_human(self, messages: list[Any]) -> list[Any]:
        """Return the messages belonging to the current user turn."""
        for index in range(len(messages) - 1, -1, -1):
            if getattr(messages[index], "type", None) == "human":
                return messages[index + 1 :]
        return list(messages)

    def _nudge_already_sent(self, messages: list[Any]) -> bool:
        """Report whether this turn already carries the corrective nudge."""
        return any(
            getattr(message, "type", None) == "system"
            and self._flatten(getattr(message, "content", None)) == READ_NUDGE
            for message in messages
        )

    def _tool_names(self, messages: list[Any]) -> list[str]:
        """Collect every tool call name issued during the turn."""
        names: list[str] = []
        for message in messages:
            for call in getattr(message, "tool_calls", None) or []:
                name = call.get("name") if isinstance(call, dict) else None
                if name:
                    names.append(str(name))
        return names

    def _is_substantive(self, content: Any) -> bool:
        """Report whether the answer is long enough or carries code to need grounding."""
        text = self._flatten(content)
        return len(text) >= SUBSTANTIVE_CHARS or CODE_FENCE in text

    def _flatten(self, content: Any) -> str:
        """Flatten string or content-block message content into plain text."""
        if isinstance(content, list):
            return "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        return str(content or "")


__all__ = ["DocsReadGateMiddleware", "READ_NUDGE", "SUBSTANTIVE_CHARS"]
