"""Retrieval enforcement: block technical answers that skipped the docs.

The docs agent prompt mandates "NEVER answer from memory", but prose alone lets
the model ship fabricated API surface whenever it judges retrieval unnecessary.
This middleware enforces the rule at runtime: a draft final answer containing a
code block or a LangChain-ecosystem symbol is rejected and the model is routed
back with an instruction to search and read the docs first.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)

#: Tools that count as an actual documentation / support-article read or search.
RETRIEVAL_TOOLS = frozenset(
    {
        "search_docs_by_lang_chain",
        "query_docs_filesystem_docs_by_lang_chain",
        "search_support_articles",
        "get_support_article_content",
        "fetch_langchain_pricing",
    }
)

#: Fenced code block, or an ecosystem symbol the answer must not assert from memory.
TECHNICAL_CLAIM_PATTERN = re.compile(
    r"```"
    r"|create_deep_agent"
    r"|create_react_agent"
    r"|create_agent"
    r"|\w*Middleware\b"
    r"|stream_mode"
    r"|langchain_\w+"
    r"|langgraph\.\w+"
    r"|JsonOutputParser"
    r"|ChatOpenAI"
)

FORCE_RETRIEVAL_INSTRUCTION = (
    "Your draft answer contains a code snippet or a LangChain-ecosystem symbol, but you "
    "performed no documentation lookup on this turn. Do not send it. Call "
    "search_docs_by_lang_chain (and search_support_articles if account-related) now, then "
    "read the relevant pages with query_docs_filesystem_docs_by_lang_chain before "
    "answering. If the docs do not confirm a claim, state that you could not find "
    "documentation for it."
)


class RetrievalEnforcementMiddleware(AgentMiddleware):
    """Reject a final answer making technical claims with no retrieval this turn."""

    def __init__(self, max_forced_retries: int = 1) -> None:
        """Initialize with a cap on forced retrieval rounds per turn."""
        super().__init__()
        self.max_forced_retries = max_forced_retries

    @hook_config(can_jump_to=["model"])
    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Force a docs read when the draft answer makes ungrounded technical claims."""
        messages = state.get("messages", [])
        if not messages:
            return None

        draft = messages[-1]
        # A message that still has tool calls is not a final answer yet.
        if getattr(draft, "type", None) != "ai" or getattr(draft, "tool_calls", None):
            return None
        if not TECHNICAL_CLAIM_PATTERN.search(_as_text(draft.content)):
            return None
        if self._retrieved_this_turn(messages):
            return None

        forced = sum(
            1
            for message in messages
            if getattr(message, "type", None) == "human"
            and _as_text(message.content) == FORCE_RETRIEVAL_INSTRUCTION
        )
        if forced >= self.max_forced_retries:
            logger.warning(
                "Retrieval enforcement exhausted after %d forced round(s); allowing "
                "ungrounded technical answer through.",
                forced,
            )
            return None

        logger.info("Blocking ungrounded technical answer; forcing a retrieval round.")
        return {
            "messages": [HumanMessage(content=FORCE_RETRIEVAL_INSTRUCTION)],
            "jump_to": "model",
        }

    def _retrieved_this_turn(self, messages: list[Any]) -> bool:
        """Report whether a retrieval tool ran since the latest real user message."""
        for message in reversed(messages[:-1]):
            if (
                getattr(message, "type", None) == "human"
                and _as_text(message.content) != FORCE_RETRIEVAL_INSTRUCTION
            ):
                return False
            if getattr(message, "name", None) in RETRIEVAL_TOOLS:
                return True
            for call in getattr(message, "tool_calls", None) or ():
                name = call.get("name") if isinstance(call, dict) else None
                if name in RETRIEVAL_TOOLS:
                    return True
        return False


def _as_text(content: Any) -> str:
    """Flatten string or content-block message content into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text", "")))
        return "\n".join(parts)
    return str(content)


__all__ = [
    "FORCE_RETRIEVAL_INSTRUCTION",
    "RETRIEVAL_TOOLS",
    "RetrievalEnforcementMiddleware",
]
