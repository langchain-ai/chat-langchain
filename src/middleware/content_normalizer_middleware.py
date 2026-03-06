"""Middleware that normalizes Anthropic signed content blocks to plain strings."""
import logging
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)


def _extract_text(content: Any) -> str:
    """Extract plain text from a message content value.

    Handles three cases:
    - ``str``: returned as-is.
    - ``list``: each element that is a bare string or a dict with
      ``"type": "text"`` contributes its text; other block types are skipped.
    - Anything else: converted with ``str()``.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


class ContentNormalizerMiddleware(AgentMiddleware[AgentState]):
    """Normalize Anthropic signed content blocks to plain strings.

    Claude models (via ``langchain-anthropic``) sometimes return content as a
    list of content blocks that carry ``extras.signature`` metadata (related to
    prompt caching / extended thinking).  Downstream code that calls
    ``message.content`` expecting a ``str`` would otherwise receive the raw
    list representation.

    This middleware runs *after* every model call and replaces any
    ``AIMessage.content`` that is a list with the concatenated plain text
    extracted from its ``"text"`` blocks.
    """

    async def aafter_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Normalize list content on AI messages after each model call."""
        messages = state.get("messages", [])
        if not messages:
            return None

        updated: list[AIMessage] = []
        changed = False

        for msg in messages:
            if isinstance(msg, AIMessage) and isinstance(msg.content, list):
                normalized = _extract_text(msg.content)
                logger.debug(
                    "ContentNormalizerMiddleware: normalised list content "
                    "(%d blocks) → %d chars",
                    len(msg.content),
                    len(normalized),
                )
                # Build a new AIMessage preserving all other fields
                new_msg = msg.model_copy(update={"content": normalized})
                updated.append(new_msg)
                changed = True
            else:
                updated.append(msg)

        if not changed:
            return None

        return {"messages": updated}


__all__ = ["ContentNormalizerMiddleware", "_extract_text"]
