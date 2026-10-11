"""Repair non-existent OpenAI model IDs in generated answers.

The docs agent grounds its answers in retrieved doc pages, which can contain
illustrative or placeholder model identifiers (e.g. ``gpt-5.5``). Reproduced
verbatim, these read as valid copy-paste code even though the model does not
exist. This ``after_model`` guard rewrites those IDs to a currently-supported
model and appends a short note so users know the substitution happened.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from src.utils.model_id_validation import (
    SUPPORTED_MODEL_SUBSTITUTE,
    contains_invalid_model_id,
    find_invalid_model_ids,
    substitute_invalid_model_ids,
)

logger = logging.getLogger(__name__)

_SUBSTITUTION_NOTE = (
    "\n\n> Note: the original documentation example referenced a non-existent "
    f"model identifier, which was replaced with `{SUPPORTED_MODEL_SUBSTITUTE}`. "
    "Substitute the currently-supported model that fits your use case."
)


class ModelValidationMiddleware(AgentMiddleware):
    """Substitute non-existent model IDs in the final answer's code."""

    def after_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Rewrite invalid model IDs in the latest AI answer, if any."""
        messages = state.get("messages", [])
        if not messages:
            return None

        message = messages[-1]
        if not isinstance(message, AIMessage) or message.tool_calls:
            return None

        content = message.content
        if not isinstance(content, str) or not contains_invalid_model_id(content):
            return None

        flagged = find_invalid_model_ids(content)
        repaired = substitute_invalid_model_ids(content) + _SUBSTITUTION_NOTE
        logger.warning(
            "Replaced non-existent model IDs in generated answer: %s",
            sorted(set(flagged)),
        )
        message.content = repaired
        return {"messages": [message]}


__all__ = ["ModelValidationMiddleware"]
