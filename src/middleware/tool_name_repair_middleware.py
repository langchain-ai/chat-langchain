"""Repair unambiguous tool names and stop repeated invalid tool calls."""

import json
import re
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import replace

import langsmith as ls
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)

_FALLBACK = "I couldn't retrieve the documentation needed to answer. Please try again."


def _key(call: ToolCall) -> tuple[str, str]:
    """Identify a tool call independently of its ID and argument ordering."""
    return call["name"], json.dumps(call["args"], sort_keys=True)


def _failed_calls(messages: list) -> Counter:
    """Count rejected calls since the latest user message."""
    calls: dict[str | None, ToolCall] = {}
    failures: Counter = Counter()
    for message in messages:
        if isinstance(message, HumanMessage):
            calls.clear()
            failures.clear()
        elif isinstance(message, AIMessage):
            calls.update({call["id"]: call for call in message.tool_calls})
        elif (
            isinstance(message, ToolMessage)
            and "is not a valid tool" in message.text
            and message.tool_call_id in calls
        ):
            failures[_key(calls.pop(message.tool_call_id))] += 1
    return failures


def _repair_name(name: str, names: set[str]) -> str:
    """Resolve a malformed name only when the bound match is unambiguous."""
    if name in names:
        return name
    candidates = {name, re.split(r"[\s=]", name, maxsplit=1)[0]}
    candidates.update(
        "langchainDocs_" + candidate[len("langgraphDocs_") :]
        for candidate in list(candidates)
        if candidate.startswith("langgraphDocs_")
    )
    matches = candidates & names
    if not matches:
        suffixes = {
            candidate.split("_", 1)[1] for candidate in candidates if "_" in candidate
        }
        matches = {
            bound
            for bound in names
            if "_" in bound and bound.split("_", 1)[1] in suffixes
        }
    return next(iter(matches)) if len(matches) == 1 else name


class ToolNameRepairMiddleware(AgentMiddleware):
    """Repair tool names and finish without tools after two identical rejections."""

    def _repair(self, request: ModelRequest, response: ModelResponse) -> ModelResponse:
        names: set[str] = set()
        for tool in request.tools:
            name = (
                tool.get("name") or tool.get("function", {}).get("name")
                if isinstance(tool, dict)
                else tool.name
            )
            if isinstance(name, str):
                names.add(name)
        repaired = False
        messages = []
        for message in response.result:
            if isinstance(message, AIMessage) and message.tool_calls:
                calls = []
                for call in message.tool_calls:
                    name = _repair_name(call["name"], names)
                    repaired |= name != call["name"]
                    calls.append({**call, "name": name})
                message = message.model_copy(update={"tool_calls": calls})
            messages.append(message)
        if repaired:
            run = ls.get_current_run_tree()
            if run is not None:
                run.metadata["tool_name_repaired"] = True
        return replace(response, result=messages)

    def _needs_final(self, request: ModelRequest, response: ModelResponse) -> bool:
        failures = _failed_calls(request.messages)
        return any(
            failures[_key(call)] >= 2
            for message in response.result
            if isinstance(message, AIMessage)
            for call in message.tool_calls
        )

    def _final_request(self, request: ModelRequest) -> ModelRequest:
        content = request.system_message.content if request.system_message else ""
        blocks: list[str | dict] = (
            [{"type": "text", "text": content}]
            if isinstance(content, str)
            else list(content)
        )
        blocks.append(
            {
                "type": "text",
                "text": "Documentation tool calls repeatedly failed. Answer using the results already gathered; if insufficient, explain that you could not retrieve the needed documentation.",
            }
        )
        return request.override(
            tools=[],
            tool_choice=None,
            system_message=SystemMessage(content=blocks),
        )

    def _final_response(self, response: ModelResponse) -> ModelResponse:
        messages = [
            message.model_copy(
                update={
                    "content": message.text.strip() or _FALLBACK,
                    "tool_calls": [],
                    "invalid_tool_calls": [],
                    "additional_kwargs": {},
                }
            )
            if isinstance(message, AIMessage)
            else message
            for message in response.result
        ]
        return replace(response, result=messages)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Repair calls before execution or make a final tool-free model call."""
        response = self._repair(request, handler(request))
        if self._needs_final(request, response):
            return self._final_response(handler(self._final_request(request)))
        return response

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Repair calls before execution or make a final tool-free model call."""
        response = self._repair(request, await handler(request))
        if self._needs_final(request, response):
            return self._final_response(await handler(self._final_request(request)))
        return response
