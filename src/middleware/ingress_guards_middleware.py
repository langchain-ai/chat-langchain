"""Ingress guards: input caps and stopped requests on Managed Deep Agents.

Input caps were previously enforced in ``src/api/auth.py`` (``validate_inputs``).
Under MDA, identity/thread scoping is declared in ``identity.py``; this
middleware caps oversized user input and supersedes unanswered requests.

Trace metadata (prompt provenance, ``LANGSMITH_AGENT_VERSION``, ``source_type``)
is applied at agent compile time via ``define_deep_agent(metadata=...)`` in
``agent.py`` — nested ``before_agent`` spans cannot reliably update the
LangSmith root run. Git-linked host fields (``LANGSMITH_LANGGRAPH_GIT_*``) are
not synthesized; archive deploys use ``LANGSMITH_HOST_REVISION_ID`` /
``LANGSMITH_AGENT_VERSION`` instead.
"""

from __future__ import annotations

import json
from typing import Any, cast

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from langgraph.runtime import Runtime

#: Upper bound on user-provided text, matching the previous ``MAX_MESSAGE_CHARS``.
MAX_MESSAGE_CHARS = 50_000


class IngressGuardsMiddleware(AgentMiddleware):
    """Cap user input and supersede stopped requests at agent ingress."""

    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Cap the latest user message and replace unanswered requests with context."""
        messages: list[Any] = state.get("messages", [])
        updates = []
        for message in reversed(messages):
            if getattr(message, "type", None) == "human":
                capped = self._truncate_content(message.content)
                if capped is not message.content:
                    updates.append(message.model_copy(update={"content": capped}))
                break

        if (
            not messages
            or not isinstance(messages[-1], HumanMessage)
            or (
                len(messages) > 1 and isinstance(messages[-2], (AIMessage, ToolMessage))
            )
        ):
            return {"messages": updates} if updates else None

        last_ai_index = next(
            (
                index
                for index in range(len(messages) - 2, -1, -1)
                if isinstance(messages[index], AIMessage)
            ),
            -1,
        )
        stopped_messages = [
            message
            for message in messages[last_ai_index + 1 : -1]
            if isinstance(message, HumanMessage)
        ]
        if not stopped_messages:
            return {"messages": updates} if updates else None

        removals: list[Any] = [
            RemoveMessage(id=cast(str, message.id)) for message in stopped_messages
        ]
        remaining = cast(list[Any], add_messages(messages, removals))
        if updates:
            remaining[-1] = updates[0]
        quoted_requests = "\n".join(
            json.dumps(message.text[:200], ensure_ascii=False)
            for message in stopped_messages
        )
        note = HumanMessage(
            content=(
                "The user stopped these earlier requests (quoted for context only):\n"
                f"{quoted_requests}\n"
                "Answer only the latest user message unless it explicitly refers "
                "back to these stopped requests."
            )
        )
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *remaining[:-1],
                note,
                remaining[-1],
            ]
        }

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
