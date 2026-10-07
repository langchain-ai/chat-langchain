"""Ingress guards: input caps and current-turn focus for Chat LangChain.

These were previously enforced in ``src/api/auth.py`` (``validate_inputs``).
Under MDA, identity/thread scoping is declared in ``identity.py``; this
middleware caps oversized user input and marks superseded questions for the model.

Trace metadata (prompt provenance, ``LANGSMITH_AGENT_VERSION``, ``source_type``)
is applied at agent compile time via ``define_deep_agent(metadata=...)`` in
``agent.py`` — nested ``before_agent`` spans cannot reliably update the
LangSmith root run. Git-linked host fields (``LANGSMITH_LANGGRAPH_GIT_*``) are
not synthesized; archive deploys use ``LANGSMITH_HOST_REVISION_ID`` /
``LANGSMITH_AGENT_VERSION`` instead.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.middleware.types import ModelCallResult, ModelRequest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

#: Upper bound on user-provided text, matching the previous ``MAX_MESSAGE_CHARS``.
MAX_MESSAGE_CHARS = 50_000
_SUPERSEDED_MESSAGE_PREFIX = (
    "[Earlier message with no reply - superseded by the user's next message] "
)


class IngressGuardsMiddleware(AgentMiddleware):
    """Cap user input and mark superseded questions in model requests only."""

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelCallResult],
    ) -> ModelCallResult:
        """Mark unanswered earlier questions without changing thread state."""
        return handler(self._mark_superseded_messages(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelCallResult]],
    ) -> ModelCallResult:
        """Mark unanswered earlier questions without changing thread state."""
        return await handler(self._mark_superseded_messages(request))

    def _mark_superseded_messages(self, request: ModelRequest) -> ModelRequest:
        """Copy human messages preceding the latest human without an AI reply."""
        messages = list(request.messages)
        latest_human_index = next(
            (
                index
                for index in range(len(messages) - 1, -1, -1)
                if isinstance(messages[index], HumanMessage)
            ),
            None,
        )
        if latest_human_index is None:
            return request

        changed = False
        for index in range(latest_human_index - 1, -1, -1):
            message = messages[index]
            if isinstance(message, AIMessage):
                break
            if isinstance(message, HumanMessage):
                content = (
                    _SUPERSEDED_MESSAGE_PREFIX + message.content
                    if isinstance(message.content, str)
                    else [
                        {"type": "text", "text": _SUPERSEDED_MESSAGE_PREFIX},
                        *message.content,
                    ]
                )
                messages[index] = message.model_copy(update={"content": content})
                changed = True
        return request.override(messages=messages) if changed else request

    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Truncate the latest user message when it exceeds the size cap."""
        messages = state.get("messages", [])
        for message in reversed(messages):
            if getattr(message, "type", None) == "human":
                capped = self._truncate_content(message.content)
                if capped is not message.content:
                    # Same id => the messages reducer overwrites in place.
                    message.content = capped
                    return {"messages": [message]}
                break
        return None

    def _truncate_content(self, content: Any) -> Any:
        """Trim user text to the cap while preserving non-text content blocks."""
        if isinstance(content, str):
            return (
                content[:MAX_MESSAGE_CHARS]
                if len(content) > MAX_MESSAGE_CHARS
                else content
            )

        if not isinstance(content, list):
            return content

        remaining = MAX_MESSAGE_CHARS
        changed = False
        truncated: list[Any] = []
        for block in content:
            if isinstance(block, str):
                text = block[:remaining]
                changed = changed or len(text) != len(block)
                truncated.append(text)
                remaining -= len(text)
            elif (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ):
                text = block["text"][:remaining]
                changed = changed or len(text) != len(block["text"])
                truncated.append({**block, "text": text})
                remaining -= len(text)
            else:
                truncated.append(block)
        return truncated if changed else content


__all__ = ["IngressGuardsMiddleware", "MAX_MESSAGE_CHARS"]
