"""Server-side wall-clock limits for agent runs."""

import asyncio
import time
from typing import Annotated, Any

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
    hook_config,
)
from langchain.agents.middleware.types import PrivateStateAttr
from langchain_core.messages import AIMessage, ToolMessage
from typing_extensions import NotRequired


class ExecutionTimeoutState(AgentState):
    """Private state used to track the current run deadline."""

    execution_started_at: NotRequired[Annotated[float, PrivateStateAttr]]


class ExecutionTimeoutMiddleware(AgentMiddleware[ExecutionTimeoutState]):
    """Stop an agent run after its configured wall-clock deadline."""

    state_schema = ExecutionTimeoutState

    def __init__(self, timeout_seconds: float) -> None:
        """Initialize the timeout middleware."""
        super().__init__()
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.timeout_seconds = timeout_seconds

    def before_agent(self, state: ExecutionTimeoutState, runtime) -> dict[str, Any]:
        """Start the wall-clock deadline for a run."""
        return {"execution_started_at": time.monotonic()}

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: ExecutionTimeoutState, runtime) -> dict[str, Any] | None:
        """Stop before a model call when the deadline has expired."""
        if self._remaining_seconds(state) <= 0:
            return self._timeout_update()
        return None

    @hook_config(can_jump_to=["end"])
    async def abefore_model(
        self, state: ExecutionTimeoutState, runtime
    ) -> dict[str, Any] | None:
        """Stop before an async model call when the deadline has expired."""
        if self._remaining_seconds(state) <= 0:
            return self._timeout_update()
        return None

    def wrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
        """Apply the deadline to synchronous model calls."""
        if self._remaining_seconds(request.state) <= 0:
            return self._timeout_model_response()
        return handler(request)

    async def awrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
        """Apply the deadline to asynchronous model calls."""
        remaining = self._remaining_seconds(request.state)
        if remaining <= 0:
            return self._timeout_model_response()
        try:
            async with asyncio.timeout(remaining):
                return await handler(request)
        except TimeoutError:
            return self._timeout_model_response()

    def wrap_tool_call(self, request: ToolCallRequest, handler) -> ToolMessage:
        """Apply the deadline to synchronous tool calls."""
        if self._remaining_seconds(request.state) <= 0:
            return self._timeout_tool_message(request)
        return handler(request)

    async def awrap_tool_call(self, request: ToolCallRequest, handler) -> ToolMessage:
        """Apply the deadline to asynchronous tool calls."""
        remaining = self._remaining_seconds(request.state)
        if remaining <= 0:
            return self._timeout_tool_message(request)
        try:
            async with asyncio.timeout(remaining):
                return await handler(request)
        except TimeoutError:
            return self._timeout_tool_message(request)

    def _remaining_seconds(self, state: ExecutionTimeoutState) -> float:
        started_at = state.get("execution_started_at")
        if started_at is None:
            return self.timeout_seconds
        return self.timeout_seconds - (time.monotonic() - started_at)

    def _timeout_update(self) -> dict[str, Any]:
        return {"jump_to": "end", "messages": [self._timeout_message()]}

    def _timeout_model_response(self) -> ModelResponse:
        return ModelResponse(result=[self._timeout_message()])

    def _timeout_tool_message(self, request: ToolCallRequest) -> ToolMessage:
        return ToolMessage(
            content=self._timeout_message().content,
            name=request.tool_call.get("name", "tool"),
            tool_call_id=request.tool_call.get("id", ""),
        )

    def _timeout_message(self) -> AIMessage:
        return AIMessage(content="The response timed out before it could be completed.")
