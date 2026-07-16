"""Terminal-answer invariant: the agent must end on a prose AIMessage.

The docs_agent graph could otherwise terminate with a non-answer as the last
user-facing message — an unexecuted tool-call structure, a model turn that
emitted only ``tool_calls``, an empty/whitespace turn, or (in pathological
cases) the ``GuardrailsMiddleware`` classifier decision JSON
(``{"decision": ..., "explanation": ...}``). This ``after_model`` hook forces
one more synthesis turn until the last message is natural-language prose, and
emits a safe fallback answer if synthesis cannot converge.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime
from typing_extensions import NotRequired

logger = logging.getLogger(__name__)

#: Bound on forced re-synthesis turns before falling back to a safe answer.
MAX_SYNTHESIS_RETRIES = 2

FALLBACK_ANSWER = (
    "I ran into a problem composing a final answer. Could you rephrase or "
    "resend your question so I can try again?"
)


class TerminalAnswerState(AgentState):
    """State schema tracking forced synthesis retries."""

    synthesis_retries: NotRequired[int]


def _looks_like_decision_json(text: str) -> bool:
    """Return whether text parses as the guardrails classifier decision object."""
    stripped = text.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return False
    try:
        parsed = json.loads(stripped)
    except (ValueError, TypeError):
        return False
    return isinstance(parsed, dict) and "decision" in parsed


def _content_is_prose(content: Any) -> bool:
    """Return whether message content is non-empty natural-language text."""
    if isinstance(content, str):
        text = content.strip()
        if not text:
            return False
        if _looks_like_decision_json(text):
            return False
        return True

    if isinstance(content, list):
        has_text = False
        for block in content:
            if isinstance(block, str):
                if block.strip():
                    has_text = True
                continue
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type in ("tool_call", "non_standard"):
                return False
            if block_type == "text" and isinstance(block.get("text"), str):
                if _looks_like_decision_json(block["text"]):
                    return False
                if block["text"].strip():
                    has_text = True
        return has_text

    return False


class TerminalAnswerMiddleware(AgentMiddleware[TerminalAnswerState]):
    """Force the agent to end on a synthesized prose AIMessage."""

    state_schema = TerminalAnswerState

    @hook_config(can_jump_to=["model", "end"])
    def after_model(
        self, state: TerminalAnswerState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Re-run synthesis (or emit a fallback) until the last message is prose."""
        messages = state.get("messages", [])
        if not messages:
            return None

        last = messages[-1]
        if not isinstance(last, AIMessage):
            return None

        # A turn with pending tool calls routes to tools normally; let it run.
        if getattr(last, "tool_calls", None):
            return None

        if _content_is_prose(last.content):
            return None

        retries = state.get("synthesis_retries", 0)
        if retries >= MAX_SYNTHESIS_RETRIES:
            logger.warning(
                "Terminal synthesis did not converge after %s retries; "
                "emitting fallback answer.",
                retries,
            )
            return {
                "messages": [AIMessage(content=FALLBACK_ANSWER)],
                "synthesis_retries": 0,
            }

        logger.info(
            "Last message is not prose (retry %s/%s); forcing synthesis turn.",
            retries + 1,
            MAX_SYNTHESIS_RETRIES,
        )
        return {"synthesis_retries": retries + 1, "jump_to": "model"}


__all__ = [
    "TerminalAnswerMiddleware",
    "TerminalAnswerState",
    "MAX_SYNTHESIS_RETRIES",
    "FALLBACK_ANSWER",
]
