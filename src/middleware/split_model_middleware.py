"""Experiment: route the final answer to a stronger model.

Tool-calling rounds run on the agent's default (cheap) model. When that model
returns a reply with no tool calls (i.e. it is ready to answer), the same request
is re-run on a stronger model with tool_choice="none", so it must write the final
answer from the context gathered so far. The cheap draft is discarded.
(Letting the strong model call tools made it re-open the search loop: 18 model
calls on one smoke question.)
"""

import logging
import os
from typing import Any, Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)


def _last_ai(response: Any) -> AIMessage | None:
    if isinstance(response, AIMessage):
        return response
    for msg in reversed(getattr(response, "result", None) or []):
        if isinstance(msg, AIMessage):
            return msg
    return None


def _is_final(response: Any) -> bool:
    ai = _last_ai(response)
    return ai is not None and not ai.tool_calls


class SplitModelMiddleware(AgentMiddleware):
    """Re-run the final (no tool call) turn on `final_model`."""

    def __init__(self, final_model: Any) -> None:
        super().__init__()
        self.final_model = final_model

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | AIMessage:
        response = handler(request)
        if not _is_final(response):
            return response
        logger.info("split-model: final turn -> %s", type(self.final_model).__name__)
        try:
            return handler(request.override(model=self.final_model, tool_choice="none"))
        except Exception:
            logger.exception("split-model: final model failed, keeping draft")
            return response

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | AIMessage:
        response = await handler(request)
        if not _is_final(response):
            return response
        logger.info("split-model: final turn -> %s", type(self.final_model).__name__)
        try:
            return await handler(request.override(model=self.final_model, tool_choice="none"))
        except Exception:
            logger.exception("split-model: final model failed, keeping draft")
            return response


def build_final_model() -> Any:
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=os.getenv("SPLIT_FINAL_MODEL", "gemini-3.8-flash"),
        thinking_level=os.getenv("SPLIT_FINAL_THINKING", "low"),
    )
