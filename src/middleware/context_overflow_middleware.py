"""Hard context-window guard around the model call.

The ``CustomSummarizationMiddleware`` only summarizes at a fixed ~130K-token
trigger, so a single turn that appends one oversized tool result (raw HTML from
a URL fetch, or a big batch of ``grep``/``read_file`` outputs) can push the
assembled prompt past the model window in one step. The model call then fails
with a ``context_length_exceeded`` 400 (``OpenAIContextOverflowError``) that
propagates unhandled and ends the root run with null outputs and no deliverable.

This middleware runs innermost, immediately around the model invocation, and:

1. Trims the assembled prompt down to a window-budget-derived token limit before
   the call, compressing the largest/oldest tool results first.
2. Catches ``context_length_exceeded`` 400s, retries once after an aggressive
   trim, and — if it still cannot fit — terminates gracefully by returning an
   ``AIMessage`` that explains the run could not complete, so the root run ends
   with a real deliverable instead of null outputs.
"""

from __future__ import annotations

import logging
from typing import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately

logger = logging.getLogger(__name__)

#: Fraction of the model context window reserved for the completion and for
#: token-estimate slack. The assembled prompt is trimmed to the remaining share.
DEFAULT_INPUT_BUDGET_FRACTION = 0.7

#: Window budget assumed when the model profile does not expose a limit.
DEFAULT_MAX_INPUT_TOKENS = 120_000

#: Hard cap on a single tool result's content, applied before trimming whole
#: messages so no lone tool call can single-handedly exceed the window.
MAX_TOOL_RESULT_CHARS = 60_000

OVERFLOW_MARKERS = (
    "context_length_exceeded",
    "contextoverflow",
    "context window",
    "maximum context length",
    "too many tokens",
    "reduce the length",
)

GRACEFUL_OVERFLOW_MESSAGE = (
    "I could not complete this run because the accumulated context exceeded the "
    "model's context window even after trimming older tool results. Please narrow "
    "the request (fewer sources, a smaller scope, or more specific queries) and "
    "try again."
)


class ContextOverflowGuardMiddleware(AgentMiddleware):
    """Trim the prompt to fit the model window and recover from overflow 400s."""

    def __init__(
        self,
        *,
        input_budget_fraction: float = DEFAULT_INPUT_BUDGET_FRACTION,
        default_max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
        max_tool_result_chars: int = MAX_TOOL_RESULT_CHARS,
    ) -> None:
        """Configure the window budget and per-tool-result size cap."""
        super().__init__()
        self.input_budget_fraction = input_budget_fraction
        self.default_max_input_tokens = default_max_input_tokens
        self.max_tool_result_chars = max_tool_result_chars

    def _max_input_tokens(self, request: ModelRequest) -> int:
        profile = getattr(request.model, "profile", None)
        if isinstance(profile, dict):
            limit = profile.get("max_input_tokens")
            if isinstance(limit, int) and limit > 0:
                return limit
        return self.default_max_input_tokens

    def _budget(self, request: ModelRequest) -> int:
        return max(1, int(self._max_input_tokens(request) * self.input_budget_fraction))

    def _count(self, request: ModelRequest) -> int:
        return count_tokens_approximately([request.system_prompt, *request.messages])

    def _cap_tool_result(self, message: AnyMessage) -> None:
        """Truncate an oversized tool result in place so it cannot dominate the window."""
        if not isinstance(message, ToolMessage):
            return
        content = message.content
        if isinstance(content, str) and len(content) > self.max_tool_result_chars:
            message.content = (
                content[: self.max_tool_result_chars]
                + "\n\n[truncated: tool result exceeded size cap]"
            )

    def _trim_to_budget(
        self, request: ModelRequest, budget: int
    ) -> ModelRequest:
        """Cap oversized tool results, then drop oldest tool results until under budget."""
        messages = list(request.messages)
        for message in messages:
            self._cap_tool_result(message)

        request = request.override(messages=messages)
        if self._count(request) <= budget:
            return request

        # Drop the oldest tool results first; they are the cheapest context to lose
        # and the largest single contributor to single-turn overflow.
        kept: list[AnyMessage] = list(messages)
        for index, message in enumerate(messages):
            if self._count(request.override(messages=kept)) <= budget:
                break
            if isinstance(message, ToolMessage):
                kept[index] = ToolMessage(
                    content="[dropped: older tool result trimmed to fit context window]",
                    tool_call_id=message.tool_call_id,
                    name=getattr(message, "name", None),
                )
        return request.override(messages=kept)

    def _is_overflow(self, error: Exception) -> bool:
        if error.__class__.__name__ == "OpenAIContextOverflowError":
            return True
        text = str(error).lower()
        return any(marker in text for marker in OVERFLOW_MARKERS)

    def _graceful_result(self) -> AIMessage:
        return AIMessage(content=GRACEFUL_OVERFLOW_MESSAGE)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Trim to the window budget, invoke, and recover from overflow 400s."""
        budget = self._budget(request)
        if self._count(request) > budget:
            request = self._trim_to_budget(request, budget)

        try:
            return await handler(request)
        except Exception as error:
            if not self._is_overflow(error):
                raise
            logger.warning("Model call overflowed context window; trimming and retrying")

        # Aggressive corrective pass: halve the budget and drop older tool results.
        trimmed = self._trim_to_budget(request, max(1, budget // 2))
        try:
            return await handler(trimmed)
        except Exception as error:
            if not self._is_overflow(error):
                raise
            logger.error(
                "Model call still overflowed after trim-and-retry; "
                "ending run with a graceful overflow message"
            )
            return self._graceful_result()


__all__ = ["ContextOverflowGuardMiddleware"]
