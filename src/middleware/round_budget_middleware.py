"""Bound how many tool rounds the docs agent takes before answering.

Most answers need 1-3 tool rounds. A tail of answers keeps searching one
command at a time (search -> read -> search ...), which adds ~1.2s per round
and pushes total latency past 20s. After NUDGE_AFTER_ROUNDS rounds in the
current user turn the model is told to answer from what it has; after
STOP_AFTER_ROUNDS rounds tool calls are forbidden so the next call must answer.
"""

import logging
from collections.abc import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

NUDGE_AFTER_ROUNDS = 4
STOP_AFTER_ROUNDS = 8

NUDGE_NOTE = (
    "\n\n## Research budget\n"
    "You have already made {rounds} rounds of tool calls for this question. "
    "Answer now from what you have read. Make one more batched round only if a fact "
    "the answer cannot do without is still missing, and say plainly what you could not verify."
)
STOP_NOTE = (
    "\n\n## Research budget reached\n"
    "Tools are no longer available for this question. Write the final answer now from "
    "what you have read, and say plainly what you could not verify."
)


def count_tool_rounds(messages: list[BaseMessage]) -> int:
    """Count AI messages with tool calls since the latest human message."""
    rounds = 0
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break
        if isinstance(message, AIMessage) and message.tool_calls:
            rounds += 1
    return rounds


def _with_note(system_message: SystemMessage | None, note: str) -> SystemMessage:
    if system_message is None:
        return SystemMessage(content=note.lstrip())
    content = system_message.content
    if isinstance(content, str):
        new_content: str | list = content + note
    else:
        new_content = [*content, {"type": "text", "text": note}]
    return system_message.model_copy(update={"content": new_content})


class RoundBudgetMiddleware(AgentMiddleware):
    """Nudge, then force, a final answer after too many tool rounds."""

    def __init__(
        self,
        nudge_after: int = NUDGE_AFTER_ROUNDS,
        stop_after: int = STOP_AFTER_ROUNDS,
    ) -> None:
        super().__init__()
        self.nudge_after = nudge_after
        self.stop_after = stop_after

    def _adjust(self, request: ModelRequest) -> ModelRequest:
        rounds = count_tool_rounds(request.messages)
        logger.warning(
            "round_budget_debug rounds=%d tail=%s",
            rounds,
            [type(m).__name__ for m in request.messages[-6:]],
        )
        if rounds >= self.nudge_after:
            logger.warning(
                "round_budget rounds=%d action=%s n_messages=%d n_tools=%d",
                rounds,
                "stop" if rounds >= self.stop_after else "nudge",
                len(request.messages),
                len(request.tools),
            )
        if rounds >= self.stop_after:
            return request.override(
                system_message=_with_note(request.system_message, STOP_NOTE),
                # Tools stay bound (the platform appends MCP docs tools after this
                # middleware, so emptying the list does not remove them); forbid
                # calling them instead.
                tool_choice="none",
            )
        if rounds >= self.nudge_after:
            return request.override(
                system_message=_with_note(
                    request.system_message, NUDGE_NOTE.format(rounds=rounds)
                )
            )
        return request

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(self._adjust(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(self._adjust(request))
