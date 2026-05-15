"""Middleware that short-circuits oversized single-turn user inputs.

Users sometimes paste entire logs, transcripts, or agent outputs into a single
message. Without a size guard the agent forwards the whole payload to every LLM
call, driving single-turn prompt tokens into the hundreds of thousands and
risking context-window limits. This middleware inspects the latest
``HumanMessage`` in ``state["messages"]`` and, when its string content exceeds
``MAX_HUMAN_CHARS``, short-circuits the run with a polite refusal so no LLM
calls are made on the oversized payload.

The threshold defaults to 20000 characters (~5K tokens) and can be overridden
via the ``MAX_HUMAN_CHARS`` environment variable or by passing
``max_human_chars`` to the middleware constructor.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)

# Default threshold: ~5K tokens worth of characters. Tuned against the agent's
# typical ~50K-token total-trace baseline so normal multi-turn threads are
# unaffected.
DEFAULT_MAX_HUMAN_CHARS = 20000


def _default_threshold() -> int:
    raw = os.getenv("MAX_HUMAN_CHARS")
    if raw is None or raw == "":
        return DEFAULT_MAX_HUMAN_CHARS
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Invalid MAX_HUMAN_CHARS=%r, falling back to default %d",
            raw,
            DEFAULT_MAX_HUMAN_CHARS,
        )
        return DEFAULT_MAX_HUMAN_CHARS
    if value <= 0:
        logger.warning(
            "MAX_HUMAN_CHARS must be positive (got %d), falling back to default %d",
            value,
            DEFAULT_MAX_HUMAN_CHARS,
        )
        return DEFAULT_MAX_HUMAN_CHARS
    return value


REFUSAL_MESSAGE = (
    "Your message is too long to process in a single turn. "
    "Please paste a smaller excerpt — the specific error, stack trace, or "
    "code snippet you need help with — or share a link to the full log "
    "instead of pasting it inline."
)


def _content_length(message: HumanMessage) -> int:
    """Return the character length of a HumanMessage's text content.

    Handles both string content and the list-of-blocks form that some chat
    models emit. Non-text blocks (images, tool refs) contribute zero so we
    only guard against pasted-text payloads, not multimodal inputs.
    """
    content = message.content
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for block in content:
            if isinstance(block, str):
                total += len(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if isinstance(text, str):
                    total += len(text)
        return total
    return 0


class InputSizeMiddleware(AgentMiddleware[AgentState]):
    """Short-circuit oversized single-message user inputs with a refusal.

    Runs as a ``before_agent`` hook so the guard fires before any LLM calls
    are made. When the latest ``HumanMessage`` exceeds the configured
    character threshold the middleware emits a single ``AIMessage`` refusal
    and jumps straight to ``end``, bypassing model and tool invocation
    entirely.
    """

    def __init__(self, max_human_chars: int | None = None):
        super().__init__()
        self.max_human_chars = (
            max_human_chars if max_human_chars is not None else _default_threshold()
        )

    @hook_config(can_jump_to=["end"])
    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        messages = state.get("messages") or []
        if not messages:
            return None

        last = messages[-1]
        if not isinstance(last, HumanMessage):
            return None

        length = _content_length(last)
        if length <= self.max_human_chars:
            return None

        logger.warning(
            "Oversized user input detected (%d chars > %d). Short-circuiting "
            "with refusal to avoid forwarding the payload to the model.",
            length,
            self.max_human_chars,
        )
        return {
            "messages": [AIMessage(content=REFUSAL_MESSAGE)],
            "jump_to": "end",
        }
