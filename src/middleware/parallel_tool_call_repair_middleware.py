"""Repair provider responses that merge parallel tool calls."""

from __future__ import annotations

import json
import logging
import uuid
from functools import cache
from typing import Any, Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)

logger = logging.getLogger(__name__)


class ParallelToolCallRepairMiddleware(AgentMiddleware):
    """Restore parallel tool calls merged into a legacy function call."""

    def _tool_names(self, request: ModelRequest) -> set[str]:
        names: set[str] = set()
        for tool in request.tools:
            if isinstance(tool, dict):
                name = tool.get("name")
                function = tool.get("function")
                if not isinstance(name, str) and isinstance(function, dict):
                    name = function.get("name")
            else:
                name = getattr(tool, "name", None)
            if isinstance(name, str) and name:
                names.add(name)
        return names

    def _segment_name(self, name: str, tool_names: set[str]) -> list[str] | None:
        ordered_names = sorted(
            tool_names, key=lambda tool_name: (-len(tool_name), tool_name)
        )

        @cache
        def segment(position: int) -> tuple[tuple[str, ...], ...]:
            if position == len(name):
                return ((),)

            results: list[tuple[str, ...]] = []
            for tool_name in ordered_names:
                if not name.startswith(tool_name, position):
                    continue
                for suffix in segment(position + len(tool_name)):
                    result = (tool_name, *suffix)
                    if result not in results:
                        results.append(result)
                    if len(results) == 2:
                        return tuple(results)
            return tuple(results)

        segmentations = [result for result in segment(0) if len(result) >= 2]
        if len(segmentations) != 1:
            return None
        return list(segmentations[0])

    def _parse_arguments(self, arguments: Any) -> list[dict[str, Any]] | None:
        if not isinstance(arguments, str):
            return None

        decoder = json.JSONDecoder()
        objects: list[dict[str, Any]] = []
        position = 0
        while position < len(arguments):
            while position < len(arguments) and arguments[position].isspace():
                position += 1
            if position == len(arguments):
                break
            try:
                value, position = decoder.raw_decode(arguments, position)
            except (TypeError, ValueError):
                return None
            if not isinstance(value, dict):
                return None
            objects.append(value)
        return objects or None

    def _update_content(self, message: Any, tool_calls: list[dict[str, Any]]) -> None:
        content = getattr(message, "content", None)
        if not isinstance(content, list):
            return

        tool_block_indexes = [
            index
            for index, block in enumerate(content)
            if isinstance(block, dict)
            and block.get("type") in {"tool_call", "function_call"}
        ]
        replacement = [dict(tool_call) for tool_call in tool_calls]
        if tool_block_indexes:
            first_index = tool_block_indexes[0]
            message.content = [
                *content[:first_index],
                *replacement,
                *(
                    block
                    for index, block in enumerate(
                        content[first_index + 1 :], first_index + 1
                    )
                    if index not in tool_block_indexes
                ),
            ]
        else:
            message.content = [*content, *replacement]

    def _repair_response(
        self, request: ModelRequest, response: ModelResponse
    ) -> ModelResponse:
        tool_names = self._tool_names(request)
        for message in response.result:
            if getattr(message, "type", None) != "ai":
                continue
            function_call = getattr(message, "additional_kwargs", {}).get(
                "function_call"
            )
            if not isinstance(function_call, dict):
                continue

            merged_name = function_call.get("name")
            if not isinstance(merged_name, str) or merged_name in tool_names:
                continue

            segmented_names = self._segment_name(merged_name, tool_names)
            if segmented_names is None:
                logger.warning(
                    "Unable to uniquely segment merged tool call name %r", merged_name
                )
                continue

            arguments = self._parse_arguments(function_call.get("arguments"))
            if arguments is None or len(arguments) != len(segmented_names):
                logger.warning(
                    "Unable to repair merged tool call %r: invalid argument sequence",
                    merged_name,
                )
                continue

            if len(arguments) <= len(getattr(message, "tool_calls", [])):
                continue

            tool_calls = [
                {
                    "name": tool_name,
                    "args": args,
                    "id": f"parallel-repair-{uuid.uuid4().hex}",
                    "type": "tool_call",
                }
                for tool_name, args in zip(segmented_names, arguments, strict=True)
            ]
            message.tool_calls = tool_calls
            self._update_content(message, tool_calls)
            logger.debug("Repaired %d merged tool calls", len(tool_calls))
        return response

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Repair merged tool calls in a synchronous model response."""
        return self._repair_response(request, handler(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Repair merged tool calls in an asynchronous model response."""
        return self._repair_response(request, await handler(request))


__all__ = ["ParallelToolCallRepairMiddleware"]
