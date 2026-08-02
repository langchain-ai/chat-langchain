"""Validate emitted tool-call names against the declared toolset.

The Gemini streaming path can merge parallel function-call chunks into a single
``additional_kwargs.function_call`` whose ``name`` is several declared tool
names concatenated with no delimiter (e.g.
``search_docs_by_lang_chainsearch_support_articles``) and whose ``arguments``
is two JSON objects back to back. Every consumer that reads ``function_call``
then sees a nonexistent tool and unparseable arguments.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)

#: Tools appended at compile time by the managed runtime (MCP docs tools) and by
#: the deep-agent harness (planning/filesystem builtins), so they are not part of
#: the ``docs_agent_tools`` list but are still legal tool-call names.
RUNTIME_TOOL_NAMES = frozenset(
    {
        "search_docs_by_lang_chain",
        "query_docs_filesystem_docs_by_lang_chain",
        "write_todos",
        "write_file",
        "edit_file",
        "read_file",
        "ls",
        "grep",
    }
)


class ToolCallNameValidationMiddleware(AgentMiddleware[AgentState]):
    """Detect and repair tool-call names that concatenate several declared tools."""

    def __init__(
        self,
        declared_tools: list[Any] | None = None,
        extra_tool_names: set[str] | None = None,
    ):
        """Build the legal tool-name set from declared, runtime, and extra names."""
        super().__init__()
        names = {
            tool if isinstance(tool, str) else getattr(tool, "name", "")
            for tool in (declared_tools or [])
        }
        self.declared_tool_names = frozenset(
            {name for name in names if name}
            | set(RUNTIME_TOOL_NAMES)
            | set(extra_tool_names or set())
        )

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Validate the latest model response before tool dispatch."""
        messages = state.get("messages", [])
        if not messages:
            return None

        message = messages[-1]
        if getattr(message, "type", None) != "ai":
            return None

        changed = self._validate_tool_calls(message)
        changed = self._validate_function_call(message) or changed
        # Same id => the messages reducer overwrites in place.
        return {"messages": [message]} if changed else None

    def _validate_tool_calls(self, message: Any) -> bool:
        """Warn about structured tool-call names that are not declared tools."""
        for tool_call in getattr(message, "tool_calls", None) or []:
            name = tool_call.get("name", "")
            if name in self.declared_tool_names:
                continue
            components = self._decompose(name)
            if components:
                logger.warning(
                    "Concatenated tool-call name in structured tool_calls: %r decomposes into %s",
                    name,
                    components,
                )
            else:
                logger.warning("Unknown tool-call name in structured tool_calls: %r", name)
        return False

    def _validate_function_call(self, message: Any) -> bool:
        """Split a merged ``additional_kwargs.function_call`` into one call per tool."""
        additional_kwargs = getattr(message, "additional_kwargs", None)
        if not isinstance(additional_kwargs, dict):
            return False

        function_call = additional_kwargs.get("function_call")
        if not isinstance(function_call, dict):
            return False

        name = function_call.get("name", "")
        if name in self.declared_tool_names:
            return False

        arguments = function_call.get("arguments", "")
        components = self._decompose(name)
        if not components:
            logger.warning(
                "Unknown function_call name %r (arguments=%r); leaving tool_calls untouched",
                name,
                arguments,
            )
            return False

        logger.warning(
            "Merged parallel function_call %r decomposes into %s (arguments=%r)",
            name,
            components,
            arguments,
        )

        argument_objects = self._split_json_objects(arguments)
        if argument_objects is None or len(argument_objects) != len(components):
            return False

        split_calls = [
            {
                "id": f"call_{uuid.uuid4().hex[:16]}",
                "type": "function",
                "function": {"name": component, "arguments": argument_text},
            }
            for component, argument_text in zip(components, argument_objects, strict=True)
        ]
        additional_kwargs["function_call"] = dict(split_calls[0]["function"])
        additional_kwargs["tool_calls"] = split_calls

        if not getattr(message, "tool_calls", None):
            message.tool_calls = [
                {
                    "name": call["function"]["name"],
                    "args": json.loads(call["function"]["arguments"]),
                    "id": call["id"],
                    "type": "tool_call",
                }
                for call in split_calls
            ]
        return True

    def _decompose(self, name: str) -> list[str] | None:
        """Greedily split a name into two or more declared tool names."""
        if not name:
            return None

        components: list[str] = []
        index = 0
        while index < len(name):
            matches = [
                candidate
                for candidate in self.declared_tool_names
                if name.startswith(candidate, index)
            ]
            if not matches:
                return None
            longest = max(matches, key=len)
            components.append(longest)
            index += len(longest)
        return components if len(components) > 1 else None

    def _split_json_objects(self, arguments: str) -> list[str] | None:
        """Split concatenated JSON objects, requiring each chunk to parse as a dict."""
        if not isinstance(arguments, str):
            return None

        chunks: list[str] = []
        depth = 0
        start = -1
        in_string = False
        escaped = False
        for position, character in enumerate(arguments):
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
                continue
            if character == '"':
                in_string = True
            elif character == "{":
                if depth == 0:
                    start = position
                depth += 1
            elif character == "}":
                if depth == 0:
                    return None
                depth -= 1
                if depth == 0:
                    chunks.append(arguments[start : position + 1])
            elif depth == 0 and not character.isspace():
                return None

        if depth != 0 or len(chunks) < 2:
            return None

        for chunk in chunks:
            try:
                if not isinstance(json.loads(chunk), dict):
                    return None
            except json.JSONDecodeError:
                return None
        return chunks


__all__ = ["RUNTIME_TOOL_NAMES", "ToolCallNameValidationMiddleware"]
