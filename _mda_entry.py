from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.errors import GraphRecursionError
from managed_deepagents.runtime import compile_managed_agent

from _mda_connectors import connectors as _connectors
from agent import agent as _definition
from identity import identity as _identity

_system_prompt = Path(__file__).with_name("instructions.md").read_text()
_RECURSION_LIMIT = 96
_VALIDATION_CUTOFF_NOTE = "\n\nValidation was cut short at the safety limit; this is the best available draft."
_VALIDATION_FALLBACK = "I couldn't complete the request because validation exceeded the safety limit."


class _SafeManagedAgent:
    """Apply recursion safety and recover a user-facing draft."""

    def __init__(self, agent: Any):
        self._agent = agent

    def __getattr__(self, name: str) -> Any:
        return getattr(self._agent, name)

    async def ainvoke(self, input: Any, config: Mapping[str, Any] | None = None, **kwargs: Any) -> Any:
        try:
            return await self._agent.ainvoke(input, self._safe_config(config), **kwargs)
        except GraphRecursionError as error:
            return self._recovery(getattr(error, "state", input))

    async def astream(
        self, input: Any, config: Mapping[str, Any] | None = None, **kwargs: Any
    ) -> AsyncIterator[Any]:
        latest = input
        try:
            async for chunk in self._agent.astream(input, self._safe_config(config), **kwargs):
                latest = self._merge_state(latest, chunk)
                yield chunk
        except GraphRecursionError as error:
            yield self._recovery(getattr(error, "state", latest))

    def invoke(self, input: Any, config: Mapping[str, Any] | None = None, **kwargs: Any) -> Any:
        try:
            return self._agent.invoke(input, self._safe_config(config), **kwargs)
        except GraphRecursionError as error:
            return self._recovery(getattr(error, "state", input))

    def _safe_config(self, config: Mapping[str, Any] | None) -> dict[str, Any]:
        safe_config = dict(config or {})
        safe_config.setdefault("recursion_limit", _RECURSION_LIMIT)
        return safe_config

    def _recovery(self, state: Any) -> dict[str, list[AIMessage]]:
        draft = self._latest_draft(state)
        content = f"{draft}{_VALIDATION_CUTOFF_NOTE}" if draft else _VALIDATION_FALLBACK
        return {"messages": [AIMessage(content=content)]}

    def _latest_draft(self, state: Any) -> str:
        messages = self._messages(state)
        for message in reversed(messages):
            if (
                isinstance(message, AIMessage) or getattr(message, "type", None) == "ai"
            ) and message.content:
                content = message.content
                return content if isinstance(content, str) else str(content)
        return ""

    def _merge_state(self, previous: Any, chunk: Any) -> Any:
        if isinstance(chunk, Mapping):
            if isinstance(previous, Mapping):
                merged = dict(previous)
                merged.update(chunk)
                return merged
            return chunk
        return chunk

    def _messages(self, state: Any) -> list[BaseMessage]:
        if isinstance(state, Mapping):
            messages = state.get("messages", [])
        else:
            messages = getattr(state, "messages", [])
        return list(messages or [])


def agent(config):
    compiled = compile_managed_agent(
        _definition,
        config,
        system_prompt=_system_prompt,
        connectors=_connectors,
        identity=_identity,
    )
    return _SafeManagedAgent(compiled)
