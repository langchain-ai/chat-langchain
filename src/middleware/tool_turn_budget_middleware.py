"""Cap the tool-calling rounds the agent spends on one user message, for RL training runs.

With TOOL_TURN_BUDGET=K, once the model has made K tool-calling turns since the latest user
message, its next call gets a closing instruction (BUDGET_NOTE) asking it to answer from what it
found. If it still asks for tools, the calls are dropped and the model is asked once more with no
tools offered; if that reply still carries calls, they are dropped and its text is the answer.
Earlier turns are unchanged and the model is never told how many turns remain, so prompts match
an uncapped run until the cap is reached. agent.py adds it only when TOOL_TURN_BUDGET is set.
"""

from __future__ import annotations

import os

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

TOOL_TURN_BUDGET_ENV = "TOOL_TURN_BUDGET"
BUDGET_NOTE = (
    "You have used all the tool calls available for this question. Do not call any more tools. "
    "Answer the question now, as well as you can, from what you have already found."
)


def budget_from_env(environ=os.environ) -> int | None:
    """The tool-turn budget TOOL_TURN_BUDGET names, or None when it is unset."""
    value = environ.get(TOOL_TURN_BUDGET_ENV, "").strip()
    if not value:
        return None
    if not value.isdigit() or not 1 <= int(value) <= 50:
        raise ValueError(f"{TOOL_TURN_BUDGET_ENV} must be a whole number from 1 to 50")
    return int(value)


def tool_turns_since_user(messages: list[BaseMessage]) -> int:
    """Assistant turns with tool calls after the latest user message."""
    turns = 0
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break
        if isinstance(message, AIMessage) and message.tool_calls:
            turns += 1
    return turns


def _without_tool_calls(response: ModelResponse) -> ModelResponse:
    result = [
        m.model_copy(update={"tool_calls": [], "invalid_tool_calls": []}) if isinstance(m, AIMessage) else m
        for m in response.result
    ]
    return ModelResponse(result=result, structured_response=response.structured_response)


def _calls_tools(response: ModelResponse) -> bool:
    return any(isinstance(m, AIMessage) and m.tool_calls for m in response.result)


def _has_text(response: ModelResponse) -> bool:
    return any(isinstance(m, AIMessage) and str(m.text or "").strip() for m in response.result)


class ToolTurnBudgetMiddleware(AgentMiddleware[AgentState]):
    """After `budget` tool-calling turns on a user message, the next turn answers without tools."""

    def __init__(self, budget: int) -> None:
        super().__init__()
        if budget < 1:
            raise ValueError("the tool-turn budget must be at least 1")
        self.budget = budget

    def _closing(self, request: ModelRequest) -> ModelRequest | None:
        if tool_turns_since_user(request.messages) < self.budget:
            return None
        return request.override(messages=[*request.messages, HumanMessage(content=BUDGET_NOTE)])

    def wrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
        closing = self._closing(request)
        if closing is None:
            return handler(request)
        response = handler(closing)
        if _calls_tools(response):
            stripped = _without_tool_calls(response)
            if _has_text(stripped):
                return stripped
            response = handler(closing.override(tools=[]))
        return _without_tool_calls(response)

    async def awrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
        closing = self._closing(request)
        if closing is None:
            return await handler(request)
        response = await handler(closing)
        if _calls_tools(response):
            stripped = _without_tool_calls(response)
            if _has_text(stripped):
                return stripped
            response = await handler(closing.override(tools=[]))
        return _without_tool_calls(response)
