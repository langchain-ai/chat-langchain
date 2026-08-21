"""Replace unsupported first-person support-action claims in final answers."""

from __future__ import annotations

import re
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

SUPPORT_PORTAL_REDIRECT = (
    "I cannot take that support action. Please use the "
    "[LangChain Support Portal](https://support.langchain.com) for help."
)

_FIRST_PERSON_ACTION = re.compile(
    r"\bI\s+(?!(?:cannot|can't|can not|do not|don't)\b)"
    r"(?:(?:have|had|am|will|shall|'m|'ve|'ll)\b|can\s+help\b|"
    r"(?:created?|opened?|filed?|submitted?|escalat\w*|flag\w*|"
    r"priorit\w*|expedit\w*|forward\w*|notif\w*|sent)\b)"
    r"[^.!?\n]{0,180}\b(?:ticket\w*|escalat\w*|flag\w*|priorit\w*|"
    r"expedit\w*|forward\w*|notif\w*|support\s+(?:team|queue|operations))\b",
    re.IGNORECASE,
)


def sanitize_support_action_claims(content: str) -> str:
    """Replace first-person claims of unsupported support actions."""
    sentences = re.split(r"(?<=[.!?])\s+", content)
    return " ".join(
        SUPPORT_PORTAL_REDIRECT if _FIRST_PERSON_ACTION.search(sentence) else sentence
        for sentence in sentences
    )


class SupportClaimValidatorMiddleware(AgentMiddleware):
    """Sanitize unsupported support-action claims after agent synthesis."""

    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Sanitize the final AI message when it claims unsupported actions."""
        messages = state.get("messages", [])
        if not messages or not isinstance(messages[-1], AIMessage):
            return None

        message = messages[-1]
        if not isinstance(message.content, str):
            return None
        sanitized = sanitize_support_action_claims(message.content)
        if sanitized == message.content:
            return None
        return {"messages": [message.model_copy(update={"content": sanitized})]}


__all__ = [
    "SUPPORT_PORTAL_REDIRECT",
    "SupportClaimValidatorMiddleware",
    "sanitize_support_action_claims",
]
