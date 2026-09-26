"""Ingress guards: input caps for Chat LangChain on Managed Deep Agents.

These were previously enforced in ``src/api/auth.py`` (``validate_inputs``).
Under MDA, identity/thread scoping is declared in ``identity.py``; this
middleware only caps oversized user input.

Trace metadata (prompt provenance, ``LANGSMITH_AGENT_VERSION``, ``source_type``)
is applied at agent compile time via ``define_deep_agent(metadata=...)`` in
``agent.py`` — nested ``before_agent`` spans cannot reliably update the
LangSmith root run. Git-linked host fields (``LANGSMITH_LANGGRAPH_GIT_*``) are
not synthesized; archive deploys use ``LANGSMITH_HOST_REVISION_ID`` /
``LANGSMITH_AGENT_VERSION`` instead.
"""

from __future__ import annotations

import re
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from src.prompts.guardrails_prompts import (
    self_reference_refusal_message,
)

#: Upper bound on user-provided text, matching the previous ``MAX_MESSAGE_CHARS``.
MAX_MESSAGE_CHARS = 50_000
_SELF_REFERENCE_PATTERNS = (
    re.compile(
        r"\b(?:what|which)\s+(?:model|provider|llm)\b.*"
        r"\b(?:you|your|use|using|currently)\b"
    ),
    re.compile(
        r"\b(?:system prompt|internal instructions?|tool list|what tools|configuration)\b"
    ),
    re.compile(r"\b(?:why|what|how)\b.*\b(?:token|tokens|cost|calculator|accounting)\b"),
    re.compile(
        r"\b(?:how|what)\b.*\b(?:manage|store|handle|maintain)\b.*"
        r"\b(?:state|conversation|chat(?:ting)?)\b"
    ),
    re.compile(
        r"\b(?:research workflow|tool pipeline|workflow|pipeline)\b.*"
        r"\b(?:you|your|use|follow)\b"
    ),
)


class IngressGuardsMiddleware(AgentMiddleware):
    """Guard internal questions and cap oversized user input at agent ingress."""

    @hook_config(can_jump_to=["end"])
    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Guard internal questions and truncate oversized user input."""
        messages = state.get("messages", [])
        for message in reversed(messages):
            if getattr(message, "type", None) == "human":
                if self._is_self_reference_question(message.content):
                    return {
                        "messages": [AIMessage(content=self_reference_refusal_message)],
                        "jump_to": "end",
                    }
                capped = self._truncate_content(message.content)
                if capped is not message.content:
                    # Same id => the messages reducer overwrites in place.
                    message.content = capped
                    return {"messages": [message]}
                break
        return None

    def _is_self_reference_question(self, content: Any) -> bool:
        """Detect questions about the assistant's private implementation."""
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = " ".join(
                block
                if isinstance(block, str)
                else block.get("text", "")
                if isinstance(block, dict) and block.get("type") == "text"
                else ""
                for block in content
            )
        else:
            return False
        normalized = " ".join(text.lower().split())
        return any(pattern.search(normalized) for pattern in _SELF_REFERENCE_PATTERNS)

    def _truncate_content(self, content: Any) -> Any:
        """Trim user text to the cap while preserving non-text content blocks."""
        if isinstance(content, str):
            return content[:MAX_MESSAGE_CHARS] if len(content) > MAX_MESSAGE_CHARS else content

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
