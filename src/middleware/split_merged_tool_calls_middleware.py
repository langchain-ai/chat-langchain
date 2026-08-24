"""Repair Gemini messages that merge multiple tool calls into one call."""

from __future__ import annotations

import json
import logging
from uuid import uuid4

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)

REGISTERED_TOOL_NAMES = (
    "query_docs_filesystem_docs_by_lang_chain",
    "search_docs_by_lang_chain",
    "search_support_articles",
    "get_support_article_content",
    "fetch_langchain_pricing",
    "check_links",
)


class SplitMergedToolCallsMiddleware(AgentMiddleware):
    """Restore tool calls merged by the model adapter."""

    def __init__(self, tool_names: tuple[str, ...] = REGISTERED_TOOL_NAMES) -> None:
        """Initialize the middleware with registered tool names."""
        super().__init__()
        self.tool_names = tuple(sorted(tool_names, key=len, reverse=True))

    def after_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, list[AIMessage]] | None:
        """Split concatenated tool calls without raising on malformed data."""
        messages = state.get("messages", [])
        if not messages or not isinstance(messages[-1], AIMessage):
            return None

        message = messages[-1]
        function_call = message.additional_kwargs.get("function_call")
        if not isinstance(function_call, dict):
            return None

        name = function_call.get("name")
        arguments = function_call.get("arguments")
        if not isinstance(name, str) or not isinstance(arguments, str):
            return None

        try:
            names = self._split_names(name)
            if len(names) < 2 or len(message.tool_calls) >= len(names):
                return None
            decoded_arguments = self._decode_arguments(arguments, len(names))
        except (TypeError, ValueError, json.JSONDecodeError):
            logger.warning("Unable to split merged Gemini tool call", exc_info=True)
            return None

        missing_calls = []
        for call_name, call_args in zip(names, decoded_arguments):
            if any(
                tool_call.get("name") == call_name
                and tool_call.get("args") == call_args
                for tool_call in message.tool_calls
            ):
                continue
            missing_calls.append(
                {
                    "name": call_name,
                    "args": call_args,
                    "id": f"call_split_{uuid4().hex[:8]}",
                    "type": "tool_call",
                }
            )

        if not missing_calls:
            return None

        patched_message = message.model_copy(
            update={"tool_calls": [*message.tool_calls, *missing_calls]}
        )
        return {"messages": [patched_message]}

    def _split_names(self, merged_name: str) -> list[str]:
        names = []
        position = 0
        while position < len(merged_name):
            match = next(
                (
                    tool_name
                    for tool_name in self.tool_names
                    if merged_name.startswith(tool_name, position)
                ),
                None,
            )
            if match is None:
                raise ValueError("merged tool name contains an unknown tool")
            names.append(match)
            position += len(match)
        return names

    def _decode_arguments(self, concatenated: str, count: int) -> list[dict]:
        decoder = json.JSONDecoder()
        arguments = []
        position = 0
        while position < len(concatenated):
            while position < len(concatenated) and concatenated[position].isspace():
                position += 1
            if position == len(concatenated):
                break
            value, position = decoder.raw_decode(concatenated, position)
            if not isinstance(value, dict):
                raise ValueError("tool arguments must be JSON objects")
            arguments.append(value)
        if len(arguments) != count:
            raise ValueError("merged tool arguments do not match tool names")
        return arguments


__all__ = ["SplitMergedToolCallsMiddleware"]
