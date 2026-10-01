"""Provide a degraded answer when the graph reaches its recursion limit."""

import logging
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda
from langgraph.errors import GraphRecursionError

logger = logging.getLogger(__name__)
_FALLBACK_MESSAGE = "I could not complete this request. Please try again."
_EXCEPTION_KEY = "_graph_recursion_error"


def with_recursion_boundary(runnable: Runnable[Any, Any]) -> Runnable[Any, Any]:
    """Return a runnable that degrades gracefully on graph recursion errors."""
    return runnable.with_fallbacks(
        [RunnableLambda(_recursion_fallback)],
        exceptions_to_handle=(GraphRecursionError,),
        exception_key=_EXCEPTION_KEY,
    )


def _recursion_fallback(value: Any) -> dict[str, list[AIMessage]]:
    error = value.get(_EXCEPTION_KEY) if isinstance(value, Mapping) else None
    if isinstance(error, GraphRecursionError):
        logger.warning("Docs agent exhausted its recursion limit", exc_info=error)

    messages = value.get("messages", []) if isinstance(value, Mapping) else []
    for message in reversed(messages):
        if getattr(message, "type", None) == "ai" and _has_content(message.content):
            return {"messages": [message]}

    return {"messages": [AIMessage(content=_FALLBACK_MESSAGE)]}


def _has_content(content: Any) -> bool:
    if isinstance(content, str):
        return bool(content.strip())
    return bool(content)
