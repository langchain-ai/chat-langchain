"""Deduplicate repeated tool calls within a single trace.

The docs agent sometimes reissues the same ``(tool_name, args)`` tuple many
times in one trace; each repeat returns identical results while burning tool
latency and prompt tokens. This middleware inspects the conversation history in
``awrap_tool_call`` and short-circuits repeats: the second identical call reuses
the cached result, the third-or-later call returns a directive telling the model
to stop repeating, and an overall cap forces a final answer once too many tool
calls have accumulated.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

logger = logging.getLogger(__name__)


class ToolCallDedupMiddleware(AgentMiddleware[AgentState]):
    """Short-circuit duplicate tool calls and cap total tool calls per trace."""

    def __init__(self, repeat_directive_after: int = 2, max_tool_calls: int = 25):
        """Configure repeat-directive threshold and per-trace tool-call cap."""
        super().__init__()
        self.repeat_directive_after = repeat_directive_after
        self.max_tool_calls = max_tool_calls

    def _signature(self, name: str, args: Any) -> str:
        try:
            args_repr = json.dumps(args, sort_keys=True, default=str)
        except TypeError:
            args_repr = repr(args)
        return f"{name}::{args_repr}"

    def _messages(self, request: ToolCallRequest) -> list:
        state = request.state
        if isinstance(state, dict):
            return state.get("messages", []) or []
        return getattr(state, "messages", []) or []

    def _history(self, request: ToolCallRequest) -> tuple[dict, dict, int]:
        """Return (id->signature, id->result content, total prior tool calls)."""
        id_to_signature: dict[str, str] = {}
        id_to_result: dict[str, str] = {}
        total_calls = 0
        for message in self._messages(request):
            if isinstance(message, AIMessage):
                for call in message.tool_calls or []:
                    call_id = call.get("id")
                    if not call_id:
                        continue
                    id_to_signature[call_id] = self._signature(
                        call.get("name", ""), call.get("args", {})
                    )
                    total_calls += 1
            elif isinstance(message, ToolMessage):
                if message.tool_call_id:
                    id_to_result[message.tool_call_id] = message.content
        return id_to_signature, id_to_result, total_calls

    def _tool_message(self, request: ToolCallRequest, content: Any) -> ToolMessage:
        return ToolMessage(
            content=content,
            name=request.tool_call.get("name", "unknown_tool"),
            tool_call_id=request.tool_call.get("id", ""),
        )

    async def awrap_tool_call(self, request: ToolCallRequest, handler) -> ToolMessage | Command:
        """Reuse cached results for repeats and cap total tool calls per trace."""
        name = request.tool_call.get("name", "")
        args = request.tool_call.get("args", {})
        signature = self._signature(name, args)
        current_id = request.tool_call.get("id")

        id_to_signature, id_to_result, total_calls = self._history(request)

        if total_calls >= self.max_tool_calls:
            logger.warning(
                "Tool-call cap (%s) reached; forcing final answer", self.max_tool_calls
            )
            return self._tool_message(
                request,
                json.dumps(
                    {
                        "notice": "Tool-call limit reached.",
                        "instruction": (
                            "You have reached the maximum number of tool calls for "
                            "this conversation. Do not call any more tools. Write your "
                            "final answer using the information already retrieved."
                        ),
                    }
                ),
            )

        prior_matches = [
            call_id
            for call_id, prior_signature in id_to_signature.items()
            if prior_signature == signature and call_id != current_id
        ]
        repeat_count = len(prior_matches)

        if repeat_count == 0:
            return await handler(request)

        cached_result = next(
            (id_to_result[call_id] for call_id in reversed(prior_matches) if call_id in id_to_result),
            None,
        )

        if repeat_count >= self.repeat_directive_after:
            logger.info(
                "Tool %s called with identical args %s times; injecting stop directive",
                name,
                repeat_count + 1,
            )
            return self._tool_message(
                request,
                json.dumps(
                    {
                        "notice": "Duplicate tool call blocked.",
                        "instruction": (
                            f"You already called {name} with these arguments and got "
                            "the result above. Do not call it again — either pick a "
                            "different query, use a different tool, or write your final "
                            "answer with the information you already have."
                        ),
                        "previous_result": cached_result,
                    },
                    default=str,
                ),
            )

        logger.info(
            "Tool %s repeated with identical args; returning cached result", name
        )
        if cached_result is not None:
            return self._tool_message(request, cached_result)
        return await handler(request)


__all__ = ["ToolCallDedupMiddleware"]
