"""Validate tool names and bound invalid-tool loops before tool execution."""

import re
from collections.abc import Awaitable, Callable
from difflib import get_close_matches
from typing import Any, cast

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.config import get_config
from langgraph.runtime import Runtime
from typing_extensions import NotRequired

APOLOGY = "Sorry, I couldn't complete the documentation lookup. Please try again."
_REPAIRS = "tool_name_repairs"


class ToolNameRepairState(AgentState):
    """Keep the repair budget local to the current user turn."""

    tool_name_repairs: NotRequired[int]


class ToolNameRepairMiddleware(AgentMiddleware[ToolNameRepairState]):
    """Repair recognizable tool names and stop repeated invalid calls."""

    state_schema = ToolNameRepairState

    def before_agent(
        self, state: ToolNameRepairState, runtime: Runtime
    ) -> dict[str, Any]:
        """Reset the repair budget for each invocation."""
        return {_REPAIRS: 0}

    def after_model(
        self, state: ToolNameRepairState, runtime: Runtime
    ) -> dict[str, Any]:
        """Persist the number of invalid calls handled by the model wrapper."""
        message = state["messages"][-1]
        return {
            _REPAIRS: state.get(_REPAIRS, 0)
            + message.response_metadata.get(_REPAIRS, 0)
        }

    def _names(self, request: ModelRequest) -> list[str]:
        names = []
        for tool in request.tools:
            name = (
                tool.get("name") or tool.get("function", {}).get("name")
                if isinstance(tool, dict)
                else tool.name
            )
            if isinstance(name, str):
                names.append(name)
        return names

    def _repair(
        self, response: ModelResponse | AIMessage, names: list[str]
    ) -> tuple[ModelResponse, int]:
        result = [response] if isinstance(response, AIMessage) else response.result
        repaired = []
        invalid = 0
        for message in result:
            if not isinstance(message, AIMessage):
                repaired.append(message)
                continue
            calls = []
            for call in message.tool_calls:
                name = call["name"]
                if name not in names:
                    invalid += 1
                    if "query_docs_filesystem" in name:
                        name = "langchainDocs_query_docs_filesystem_docs_by_lang_chain"
                    elif "search_docs" in name:
                        name = "langchainDocs_search_docs_by_lang_chain"
                    else:
                        matches = get_close_matches(
                            re.split(r"[\s<]", name, maxsplit=1)[0],
                            names,
                            n=1,
                            cutoff=0.8,
                        )
                        name = matches[0] if matches else ""
                if name in names:
                    calls.append({**call, "name": name})
            repaired.append(message.model_copy(update={"tool_calls": calls}))
        return ModelResponse(
            result=repaired,
            structured_response=None
            if isinstance(response, AIMessage)
            else response.structured_response,
        ), invalid

    def _note(self, request: ModelRequest, final: bool) -> ModelRequest:
        note = (
            "Answer now using the available context without calling tools. If you cannot answer, explain that the lookup could not be completed."
            if final
            else "Use only these exact tool names: " + ", ".join(self._names(request))
        )
        content: list[str | dict[str, Any]] = []
        if request.system_message:
            original = request.system_message.content
            content.extend([original] if isinstance(original, str) else original)
        updated = request.override(
            system_message=SystemMessage(
                content=[*content, {"type": "text", "text": note}]
            )
        )
        return updated.override(tools=[], tool_choice=None) if final else updated

    def _finish(
        self, response: ModelResponse, invalid: int, final: bool = False
    ) -> ModelResponse:
        for index, message in enumerate(response.result):
            if isinstance(message, AIMessage):
                response.result[index] = message.model_copy(
                    update={
                        "tool_calls": [] if final else message.tool_calls,
                        "content": (
                            message.content if message.text.strip() else APOLOGY
                        )
                        if final
                        else message.content,
                        "response_metadata": {
                            **message.response_metadata,
                            _REPAIRS: invalid,
                        },
                    }
                )
        return response

    def _empty(self, response: ModelResponse) -> bool:
        return not any(
            isinstance(message, AIMessage)
            and (message.tool_calls or message.text.strip())
            for message in response.result
        )

    def _exhausted(self, request: ModelRequest) -> bool:
        try:
            config = get_config()
        except RuntimeError:
            return False
        # Leave enough steps for the remaining after-model hooks and graph exit.
        return (
            config["recursion_limit"]
            - config.get("metadata", {}).get("langgraph_step", 0)
            <= 6
        )

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Validate synchronous model calls with one recovery retry."""
        if self._exhausted(request):
            return ModelResponse(result=[AIMessage(content=APOLOGY)])
        invalid = 0
        for attempt in range(2):
            response, count = self._repair(handler(request), self._names(request))
            invalid += count
            if cast(int, request.state.get(_REPAIRS, 0)) + invalid >= 3:
                response, _ = self._repair(handler(self._note(request, True)), [])
                return self._finish(response, invalid, True)
            if count and self._empty(response) and attempt == 0:
                request = self._note(request, False)
                continue
            return self._finish(response, invalid, self._empty(response))
        raise AssertionError("Unreachable recovery state")

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Validate asynchronous model calls with one recovery retry."""
        if self._exhausted(request):
            return ModelResponse(result=[AIMessage(content=APOLOGY)])
        invalid = 0
        for attempt in range(2):
            response, count = self._repair(await handler(request), self._names(request))
            invalid += count
            if cast(int, request.state.get(_REPAIRS, 0)) + invalid >= 3:
                response, _ = self._repair(await handler(self._note(request, True)), [])
                return self._finish(response, invalid, True)
            if count and self._empty(response) and attempt == 0:
                request = self._note(request, False)
                continue
            return self._finish(response, invalid, self._empty(response))
        raise AssertionError("Unreachable recovery state")
