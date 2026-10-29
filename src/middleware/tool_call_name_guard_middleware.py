"""Repair malformed tool names before provider calls and checkpointing."""

import logging
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any, TypeVar, cast

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

logger = logging.getLogger(__name__)
_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
_Message = TypeVar("_Message", bound=BaseMessage)


class ToolCallNameGuardMiddleware(AgentMiddleware):
    """Keep unregistered or provider-invalid tool names out of model traffic."""

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Clean history and repair responses, retrying unresolved calls once."""
        schemas = self._tool_schemas(request)
        messages, _ = self._clean_messages(request.messages, schemas)
        request = request.override(messages=messages)
        response, unresolved = self._clean_response(handler(request), schemas)
        if unresolved:
            response, _ = self._clean_response(handler(request), schemas)
        return response

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Clean history and repair responses, retrying unresolved calls once."""
        schemas = self._tool_schemas(request)
        messages, _ = self._clean_messages(request.messages, schemas)
        request = request.override(messages=messages)
        response, unresolved = self._clean_response(await handler(request), schemas)
        if unresolved:
            response, _ = self._clean_response(await handler(request), schemas)
        return response

    def _tool_schemas(self, request: ModelRequest) -> dict[str, dict[str, Any]]:
        schemas = {}
        for tool in request.tools:
            if isinstance(tool, dict):
                definition = tool.get("function", tool)
                name = definition.get("name")
                schema = definition.get("parameters", {})
            else:
                definition = convert_to_openai_tool(tool)["function"]
                name = definition["name"]
                schema = definition.get("parameters", {})
            if isinstance(name, str) and _NAME_PATTERN.fullmatch(name):
                schemas[name] = schema
        return schemas

    def _repair_name(
        self, call: Mapping[str, Any], schemas: dict[str, dict[str, Any]]
    ) -> str | None:
        name = call["name"]
        if name in schemas:
            return name
        stripped = re.sub(r"[^A-Za-z0-9_-]", "", name)
        if stripped in schemas:
            return stripped
        for token in name.split("|"):
            if token in schemas:
                return token
        keys = set(call["args"])
        candidates = [
            tool_name
            for tool_name, schema in schemas.items()
            if "properties" in schema
            and set(schema.get("required", [])) <= keys <= set(schema["properties"])
        ]
        return candidates[0] if len(candidates) == 1 else None

    def _clean_messages(
        self, messages: Sequence[_Message], schemas: dict[str, dict[str, Any]]
    ) -> tuple[list[_Message], bool]:
        cleaned: list[_Message] = []
        dropped_ids = set()
        unresolved = False
        for index, message in enumerate(messages):
            if not isinstance(message, AIMessage):
                cleaned.append(message)
                continue
            calls = []
            replacements = {}
            changed = False
            for call in message.tool_calls:
                name = self._repair_name(call, schemas)
                replacements[call["id"]] = name
                if name != call["name"]:
                    changed = True
                    logger.warning(
                        "Tool call name guard: message %d name %r -> %r",
                        index,
                        call["name"],
                        name if name is not None else "drop",
                    )
                if name is None:
                    unresolved = True
                    dropped_ids.add(call["id"])
                else:
                    calls.append({**call, "name": name})
            if changed:
                additional_kwargs = dict(message.additional_kwargs)
                if "tool_calls" in additional_kwargs:
                    additional_kwargs["tool_calls"] = self._clean_blocks(
                        additional_kwargs["tool_calls"], replacements
                    )
                    if not additional_kwargs["tool_calls"]:
                        del additional_kwargs["tool_calls"]
                if "function_call" in additional_kwargs:
                    legacy = additional_kwargs["function_call"]
                    legacy_name = next(
                        (
                            replacements[call["id"]]
                            for call in message.tool_calls
                            if call["name"] == legacy.get("name")
                        ),
                        None,
                    )
                    if legacy_name is None:
                        del additional_kwargs["function_call"]
                    else:
                        additional_kwargs["function_call"] = {
                            **legacy,
                            "name": legacy_name,
                        }
                content = message.content
                if isinstance(content, list):
                    content = self._clean_blocks(content, replacements)
                message = cast(
                    _Message,
                    message.model_copy(
                        update={
                            "tool_calls": calls,
                            "additional_kwargs": additional_kwargs,
                            "content": content,
                        }
                    ),
                )
            cleaned.append(message)
        result = []
        for index, message in enumerate(cleaned):
            if isinstance(message, ToolMessage) and message.tool_call_id in dropped_ids:
                logger.warning(
                    "Tool call name guard: message %d drop ToolMessage for %r",
                    index,
                    message.tool_call_id,
                )
                continue
            result.append(message)
        return result, unresolved

    def _clean_blocks(
        self, blocks: list[Any], replacements: dict[str | None, str | None]
    ) -> list[Any]:
        cleaned = []
        for block in blocks:
            if not isinstance(block, dict) or block.get("id") not in replacements:
                cleaned.append(block)
                continue
            if block.get("type") not in ("function", "tool_use", "tool_call"):
                cleaned.append(block)
                continue
            name = replacements[block.get("id")]
            if name is None:
                continue
            if "function" in block:
                block = {**block, "function": {**block["function"], "name": name}}
            else:
                block = {**block, "name": name}
            cleaned.append(block)
        return cleaned

    def _clean_response(
        self, response: ModelResponse, schemas: dict[str, dict[str, Any]]
    ) -> tuple[ModelResponse, bool]:
        messages, unresolved = self._clean_messages(response.result, schemas)
        if any(
            original is not cleaned
            for original, cleaned in zip(response.result, messages)
        ) or len(messages) != len(response.result):
            response = replace(response, result=messages)
        return response, unresolved


__all__ = ["ToolCallNameGuardMiddleware"]
