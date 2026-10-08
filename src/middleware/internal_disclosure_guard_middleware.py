"""Validate buffered answers before exposing internal tool identifiers."""

import asyncio
import re
from collections.abc import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage
from langchain_core.runnables.config import ensure_config, set_config_context

from src.prompts.guardrails_prompts import internal_disclosure_refusal

_PUBLIC_TOOL_APIS = {
    "tool",
    "bind_tools",
    "ls",
    "glob",
    "grep",
    "read_file",
    "write_file",
    "edit_file",
    "execute",
    "write_todos",
    "task",
}


class InternalDisclosureGuardMiddleware(AgentMiddleware):
    """Buffer model calls and replace terminal disclosures without retrying."""

    def _validate(
        self, request: ModelRequest, response: ModelCallResult
    ) -> ModelCallResult:
        """Check terminal answer text against the effective runtime tool registry."""
        model_response = getattr(response, "model_response", response)
        messages = (
            model_response.result
            if isinstance(model_response, ModelResponse)
            else [model_response]
        )
        if any(
            isinstance(message, AIMessage) and message.tool_calls
            for message in messages
        ):
            return response
        names = set()
        for tool in request.tools:
            if isinstance(tool, dict):
                name = tool.get("name") or tool.get("function", {}).get("name")
            else:
                name = tool.name
            if name and name.lower() not in _PUBLIC_TOOL_APIS:
                names.add(name)
        if not names:
            return response
        pattern = re.compile(
            r"\b(?:" + "|".join(re.escape(name) for name in names) + r")\b",
            re.IGNORECASE | re.ASCII,
        )
        if any(
            isinstance(message, AIMessage) and pattern.search(message.text)
            for message in messages
        ):
            if isinstance(model_response, ModelResponse):
                model_response.result = [AIMessage(content=internal_disclosure_refusal)]
                model_response.structured_response = None
                return response
            return ModelResponse(
                result=[AIMessage(content=internal_disclosure_refusal)]
            )
        return response

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelCallResult],
    ) -> ModelCallResult:
        """Return only validated output from a synchronous model call."""
        config = ensure_config()
        config["callbacks"] = []
        # Child model events must remain private until the outer node emits validated output.
        with set_config_context(config) as context:
            response = context.run(handler, request)
        return self._validate(request, response)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelCallResult]],
    ) -> ModelCallResult:
        """Return only validated output from an asynchronous model call."""
        config = ensure_config()
        config["callbacks"] = []

        async def invoke() -> ModelCallResult:
            return await handler(request)

        with set_config_context(config) as context:
            response = await asyncio.create_task(invoke(), context=context)
        return self._validate(request, response)
