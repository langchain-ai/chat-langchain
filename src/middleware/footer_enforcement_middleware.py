"""Output-stage post-processor that enforces the "Relevant docs:" footer.

The docs agent is prompt-instructed to close every substantive in-scope answer
with a ``**Relevant docs:**`` footer (or a localized equivalent) and a bullet
list of documentation links. On short conversational closers and answers that
end naturally on a code block or architecture-principle sentence the model
sometimes drops the footer. This middleware runs after the agent emits its
final ``AIMessage`` and, when the footer is missing from a substantive answer,
asks the model once to append it using only URLs already retrieved in the
conversation's tool results — it never fabricates links.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)

#: Minimum length (chars) for an answer to be considered substantive.
MIN_SUBSTANTIVE_CHARS = 300

#: Footer variants that satisfy the closing-format rule (incl. localized).
FOOTER_VARIANTS = (
    "**Relevant docs:**",
    "**相关文档：**",
    "**Documentación relevante:**",
    "**Documentação relevante:**",
)

_URL_PATTERN = re.compile(r"https?://[^\s)\]\"'>]+")

_REWRITE_SYSTEM_PROMPT = (
    "You are a formatting post-processor for a LangChain documentation "
    "assistant. You will be given a final answer that is missing its required "
    "closing footer. Return the SAME answer verbatim, then append a footer: a "
    'blank line, the line "**Relevant docs:**", a blank line, and a bullet list '
    "of at least 2 documentation links in [title](url) format. Use ONLY the "
    "URLs provided below — do NOT invent, guess, or modify any URL. Do not "
    "change, summarize, or add anything else to the answer body."
)


class FooterEnforcementMiddleware(AgentMiddleware):
    """Append the "Relevant docs:" footer to substantive answers missing it."""

    def __init__(self, model: str | None = None):
        """Initialize the rewrite model used to append a missing footer."""
        super().__init__()
        if model is None:
            from src.agent.config import DEFAULT_MODEL

            model = DEFAULT_MODEL.id
        self.llm = init_chat_model(model=model, temperature=0)

    async def aafter_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Rewrite the final message to add the footer when it is missing."""
        messages = state.get("messages", [])
        if not messages:
            return None

        final = messages[-1]
        if not isinstance(final, AIMessage) or getattr(final, "tool_calls", None):
            return None

        text = self._extract_text(final.content)
        if not text or len(text) < MIN_SUBSTANTIVE_CHARS:
            return None
        if any(variant in text for variant in FOOTER_VARIANTS):
            return None

        urls = self._collect_tool_urls(messages)
        if len(urls) < 2:
            logger.warning(
                "Final answer missing footer but no documentation URLs found in "
                "tool history; passing message through unmodified."
            )
            return None

        rewritten = await self._append_footer(text, urls)
        if not rewritten or not any(v in rewritten for v in FOOTER_VARIANTS):
            return None

        # Same id => the messages reducer overwrites in place.
        final.content = rewritten
        return {"messages": [final]}

    async def _append_footer(self, text: str, urls: list[str]) -> str | None:
        """Invoke the model once to append the footer using retrieved URLs."""
        url_block = "\n".join(f"- {url}" for url in urls)
        prompt = [
            SystemMessage(content=_REWRITE_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Available documentation URLs:\n{url_block}\n\n"
                    f"Answer to append the footer to:\n{text}"
                )
            ),
        ]
        try:
            response = await self.llm.ainvoke(prompt)
        except Exception as exc:
            logger.warning("Footer rewrite failed, passing message through: %s", exc)
            return None
        return self._extract_text(response.content) or None

    def _collect_tool_urls(self, messages: list) -> list[str]:
        """Collect unique documentation URLs from prior tool-call results."""
        urls: list[str] = []
        seen: set[str] = set()
        for msg in messages:
            if not isinstance(msg, ToolMessage):
                continue
            for url in _URL_PATTERN.findall(self._extract_text(msg.content) or ""):
                cleaned = url.rstrip(".,;")
                if cleaned not in seen:
                    seen.add(cleaned)
                    urls.append(cleaned)
        return urls

    def _extract_text(self, content: Any) -> str:
        """Extract plain text from string or block-list message content."""
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""
        parts = [
            block if isinstance(block, str) else block.get("text", "")
            for block in content
            if isinstance(block, str)
            or (isinstance(block, dict) and block.get("type") == "text")
        ]
        return "".join(parts)


__all__ = [
    "FooterEnforcementMiddleware",
    "FOOTER_VARIANTS",
    "MIN_SUBSTANTIVE_CHARS",
]
