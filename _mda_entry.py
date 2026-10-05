from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.errors import GraphRecursionError
from managed_deepagents.runtime import compile_managed_agent

from _mda_connectors import connectors as _connectors
from agent import agent as _definition
from identity import identity as _identity

_system_prompt = Path(__file__).with_name("instructions.md").read_text()
_RECURSION_FALLBACK = (
    "I’m sorry, but I couldn’t complete that request. Please try again."
)


class _RecursionSafeRunnable:
    """Return a useful response when the managed graph reaches its recursion limit."""

    def __init__(self, runnable: Any):
        self._runnable = runnable

    def __getattr__(self, name: str) -> Any:
        return getattr(self._runnable, name)

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        try:
            return self._runnable.invoke(input, config=config, **kwargs)
        except GraphRecursionError:
            return self._fallback(input)

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        try:
            return await self._runnable.ainvoke(input, config=config, **kwargs)
        except GraphRecursionError:
            return self._fallback(input)

    def stream(self, input: Any, config: Any = None, **kwargs: Any) -> Iterator[Any]:
        last_answer: str | None = None
        try:
            for chunk in self._runnable.stream(input, config=config, **kwargs):
                last_answer = self._answer_from_output(chunk) or last_answer
                yield chunk
        except GraphRecursionError:
            yield self._fallback(input, last_answer)

    async def astream(
        self, input: Any, config: Any = None, **kwargs: Any
    ) -> AsyncIterator[Any]:
        last_answer: str | None = None
        try:
            async for chunk in self._runnable.astream(input, config=config, **kwargs):
                last_answer = self._answer_from_output(chunk) or last_answer
                yield chunk
        except GraphRecursionError:
            yield self._fallback(input, last_answer)

    def _fallback(self, input: Any, last_answer: str | None = None) -> dict[str, list[AIMessage]]:
        return {"messages": [AIMessage(content=last_answer or _RECURSION_FALLBACK)]}

    def _answer_from_output(self, output: Any) -> str | None:
        messages = output.get("messages", []) if isinstance(output, dict) else []
        for message in reversed(messages):
            if isinstance(message, AIMessage) and isinstance(message.content, str):
                return message.content
        return None


def agent(config):
    runnable = compile_managed_agent(
        _definition,
        config,
        system_prompt=_system_prompt,
        connectors=_connectors,
        identity=_identity,
    )
    return _RecursionSafeRunnable(runnable)
