"""Guarantee every completed run ends with a synthesized assistant answer.

The agent loop can terminate on a bare tool result (or on an AI message that
carries only tool calls / empty content). When that happens the root run has no
trailing assistant text and the user receives nothing. This middleware runs
before the graph reaches END and, if the trajectory would otherwise end without
a grounded answer, invokes the model once more to synthesize a final reply from
the retrieved documents.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.runnables import Runnable
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)


def _text_content(message: AnyMessage) -> str:
    """Return the plain-text portion of a message's content."""
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts).strip()
    return ""


def needs_final_answer(messages: list[AnyMessage]) -> bool:
    """Return True when the trajectory ends without a grounded assistant answer."""
    if not messages:
        return False
    last = messages[-1]
    last_type = getattr(last, "type", None)
    if last_type == "tool":
        return True
    if last_type == "ai":
        has_tool_calls = bool(getattr(last, "tool_calls", None))
        return has_tool_calls or not _text_content(last)
    return False


class FinalSynthesisMiddleware(AgentMiddleware[AgentState]):
    """Synthesize a final answer whenever a run would end on a tool result."""

    def __init__(self, model: Runnable | BaseLanguageModel | None = None):
        """Store the model used to synthesize the fallback answer."""
        super().__init__()
        self._model = model

    def _resolve_model(self) -> Runnable | BaseLanguageModel:
        if self._model is None:
            # Imported lazily so this module has no import-time dependency on the
            # configured model registry.
            from src.agent.config import default_model

            self._model = default_model
        return self._model

    async def aafter_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Invoke the model once more if the run ends on a tool result."""
        messages = state.get("messages", [])
        if not needs_final_answer(messages):
            return None

        logger.warning(
            "Run ended without a synthesized answer; invoking model to generate "
            "a final grounded reply from tool results."
        )
        try:
            response = await self._resolve_model().ainvoke(messages)
        except Exception:
            logger.exception("Final-answer synthesis failed")
            return None

        if getattr(response, "tool_calls", None) or not _text_content(response):
            logger.warning("Synthesis produced no usable final answer")
            return None

        if not isinstance(response, AIMessage):
            response = AIMessage(content=_text_content(response))
        return {"messages": [response]}


__all__ = ["FinalSynthesisMiddleware", "needs_final_answer"]
