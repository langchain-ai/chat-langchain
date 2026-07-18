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

import json
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

#: Upper bound on user-provided text, matching the previous ``MAX_MESSAGE_CHARS``.
MAX_MESSAGE_CHARS = 50_000

#: Keys that identify a serialized LangGraph/agent-state object rather than a
#: genuine free-text user question.
_AGENT_STATE_MARKERS = ("thread_model_call_count", "values", "off_topic_query")

_MISROUTED_REJECTION = (
    "This request could not be processed: the input appears to be internal "
    "state from another system rather than a LangChain documentation question."
)


class IngressGuardsMiddleware(AgentMiddleware):
    """Cap oversized user input at agent ingress."""

    @hook_config(can_jump_to=["end"])
    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Reject mis-routed agent state, else truncate oversized user input."""
        messages = state.get("messages", [])
        for message in reversed(messages):
            if getattr(message, "type", None) == "human":
                if self._is_misrouted_agent_state(message.content):
                    return {
                        "messages": [AIMessage(content=_MISROUTED_REJECTION)],
                        "jump_to": "end",
                    }
                capped = self._truncate_content(message.content)
                if capped is not message.content:
                    # Same id => the messages reducer overwrites in place.
                    message.content = capped
                    return {"messages": [message]}
                break
        return None

    def _is_misrouted_agent_state(self, content: Any) -> bool:
        """Detect content that is a serialized agent-state object, not a query."""
        text = content if isinstance(content, str) else None
        if text is None and isinstance(content, list):
            for block in content:
                if isinstance(block, str):
                    text = block
                    break
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text")
                    break
        if not isinstance(text, str):
            return False
        stripped = text.strip()
        if not stripped.startswith("{"):
            return False
        try:
            parsed = json.loads(stripped)
        except (ValueError, TypeError):
            return False
        if not isinstance(parsed, dict):
            return False
        inner = parsed.get("values", parsed)
        if not isinstance(inner, dict):
            inner = parsed
        has_markers = any(k in parsed or k in inner for k in _AGENT_STATE_MARKERS)
        has_messages = isinstance(inner.get("messages"), list)
        names_agent = "name" in parsed or "name" in inner
        return has_messages and (has_markers or names_agent)

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
