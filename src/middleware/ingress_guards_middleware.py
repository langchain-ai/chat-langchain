"""Ingress guards: input caps and stranded-turn repair on Managed Deep Agents.

Input caps were previously enforced in ``src/api/auth.py`` (``validate_inputs``).
Under MDA, identity/thread scoping is declared in ``identity.py``; this
middleware caps oversized user input and marks unanswered trailing turns.

Trace metadata (prompt provenance, ``LANGSMITH_AGENT_VERSION``, ``source_type``)
is applied at agent compile time via ``define_deep_agent(metadata=...)`` in
``agent.py`` — nested ``before_agent`` spans cannot reliably update the
LangSmith root run. Git-linked host fields (``LANGSMITH_LANGGRAPH_GIT_*``) are
not synthesized; archive deploys use ``LANGSMITH_HOST_REVISION_ID`` /
``LANGSMITH_AGENT_VERSION`` instead.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

#: Upper bound on user-provided text, matching the previous ``MAX_MESSAGE_CHARS``.
MAX_MESSAGE_CHARS = 50_000


class IngressGuardsMiddleware(AgentMiddleware):
    """Cap oversized user input and repair stranded turns at agent ingress."""

    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Cap the latest user message and mark preceding unanswered turns."""
        messages = list(state.get("messages", []))
        update = None
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if getattr(message, "type", None) == "human":
                capped = self._truncate_content(message.content)
                if capped is not message.content:
                    message = message.model_copy(update={"content": capped})
                    messages[index] = message
                    update = {"messages": [message]}
                break

        trailing_start = len(messages)
        while trailing_start > 0 and messages[trailing_start - 1].type == "human":
            trailing_start -= 1
        if len(messages) - trailing_start < 2:
            return update

        repaired = messages[:trailing_start]
        for message in messages[trailing_start:-1]:
            repaired.extend(
                [
                    message,
                    AIMessage(
                        content="[Previous request was stopped before an answer was produced.]",
                        id=str(uuid4()),
                    ),
                ]
            )
        repaired.append(messages[-1])
        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *repaired]}

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
