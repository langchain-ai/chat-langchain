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
from collections.abc import Collection
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, RemoveMessage, ToolMessage
from langgraph.runtime import Runtime

#: Upper bound on user-provided text, matching the previous ``MAX_MESSAGE_CHARS``.
MAX_MESSAGE_CHARS = 50_000
TOOL_CALL_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def normalize_tool_call_name(
    name: Any, registered_tool_names: Collection[str]
) -> str | None:
    """Normalize a malformed tool-call name against registered tools."""
    if not isinstance(name, str):
        return None
    if TOOL_CALL_NAME_PATTERN.fullmatch(name):
        return name

    candidates = [
        registered_name
        for registered_name in registered_tool_names
        if registered_name and registered_name in name
    ]
    if not candidates:
        return None
    return max(
        candidates, key=lambda candidate: (name.endswith(candidate), len(candidate))
    )


class IngressGuardsMiddleware(AgentMiddleware):
    """Cap oversized user input at agent ingress."""

    def __init__(self, registered_tool_names: Collection[str] = ()):
        """Configure the registered tool names used for repair."""
        super().__init__()
        self.registered_tool_names = set(registered_tool_names)

    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Truncate the latest user message when it exceeds the size cap."""
        updates: list[Any] = []
        messages = state.get("messages", [])
        for message in reversed(messages):
            if getattr(message, "type", None) == "human":
                capped = self._truncate_content(message.content)
                if capped is not message.content:
                    # Same id => the messages reducer overwrites in place.
                    message.content = capped
                    updates.append(message)
                break
        updates.extend(self._sanitize_messages(messages))
        return {"messages": updates} if updates else None

    def before_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Repair malformed tool-call names before invoking a model."""
        updates = self._sanitize_messages(state.get("messages", []))
        return {"messages": updates} if updates else None

    def _sanitize_messages(self, messages: list[Any]) -> list[Any]:
        updates: list[Any] = []
        tool_messages = {
            message.tool_call_id: message
            for message in messages
            if isinstance(message, ToolMessage) and message.tool_call_id
        }

        for message in messages:
            if not isinstance(message, AIMessage):
                continue

            repaired_tool_calls: list[Any] = []
            removed_tool_call_ids: set[str] = set()
            changed = False
            for tool_call in message.tool_calls:
                name = tool_call.get("name")
                normalized_name = normalize_tool_call_name(
                    name, self.registered_tool_names
                )
                if normalized_name is None:
                    repaired_tool_call_ids = tool_call.get("id")
                    if repaired_tool_call_ids:
                        removed_tool_call_ids.add(repaired_tool_call_ids)
                    changed = True
                    continue
                if normalized_name != name:
                    tool_call = {**tool_call, "name": normalized_name}
                    changed = True
                repaired_tool_calls.append(tool_call)

            additional_kwargs = dict(message.additional_kwargs)
            function_call = additional_kwargs.get("function_call")
            if isinstance(function_call, dict) and "name" in function_call:
                function_call_name = function_call["name"]
                normalized_name = normalize_tool_call_name(
                    function_call_name, self.registered_tool_names
                )
                if normalized_name is None:
                    del additional_kwargs["function_call"]
                elif normalized_name != function_call_name:
                    additional_kwargs["function_call"] = {
                        **function_call,
                        "name": normalized_name,
                    }
                changed = changed or normalized_name != function_call_name

            if not changed:
                continue

            updates.append(
                message.model_copy(
                    update={
                        "tool_calls": repaired_tool_calls,
                        "additional_kwargs": additional_kwargs,
                    }
                )
            )
            for tool_call_id in removed_tool_call_ids:
                paired_message = tool_messages.get(tool_call_id)
                if paired_message is not None and paired_message.id is not None:
                    updates.append(RemoveMessage(id=paired_message.id))
            for tool_call in repaired_tool_calls:
                original_name = next(
                    (
                        original_call.get("name")
                        for original_call in message.tool_calls
                        if original_call.get("id") == tool_call.get("id")
                    ),
                    None,
                )
                if tool_call.get("name") == original_name:
                    continue
                paired_message = tool_messages.get(tool_call.get("id"))
                if paired_message is not None:
                    updates.append(
                        paired_message.model_copy(update={"name": tool_call["name"]})
                    )
        return updates

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


__all__ = [
    "IngressGuardsMiddleware",
    "MAX_MESSAGE_CHARS",
    "TOOL_CALL_NAME_PATTERN",
    "normalize_tool_call_name",
]
