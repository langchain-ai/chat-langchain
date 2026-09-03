"""Guard assistant output before it reaches the user."""

from __future__ import annotations

import json
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, AgentState, hook_config
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime
from typing_extensions import NotRequired


class FinalAnswerGuardState(AgentState):
    """State used to limit malformed-answer retries to one per invocation."""

    final_answer_guard_retried: NotRequired[bool]


class FinalAnswerGuardMiddleware(AgentMiddleware[FinalAnswerGuardState]):
    """Prevent serialized tool calls and empty answers from reaching users."""

    state_schema = FinalAnswerGuardState

    @staticmethod
    def _normalise_content(content: Any) -> str:
        """Convert supported message content blocks to text or JSON."""
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            if isinstance(content.get("text"), str):
                return content["text"]
            return json.dumps(content, sort_keys=True, default=str)
        if isinstance(content, list):
            text_parts = []
            structured_parts = []
            for block in content:
                if isinstance(block, str):
                    text_parts.append(block)
                elif isinstance(block, dict) and isinstance(block.get("text"), str):
                    text_parts.append(block["text"])
                else:
                    structured_parts.append(block)
            text = "".join(text_parts)
            if structured_parts and not text.strip():
                return json.dumps(content, sort_keys=True, default=str)
            return text + "".join(
                json.dumps(block, sort_keys=True, default=str)
                for block in structured_parts
            )
        return json.dumps(content, sort_keys=True, default=str)

    @classmethod
    def _is_tool_call_residue(cls, content: Any) -> bool:
        """Check whether content is solely a serialized tool-call payload."""
        normalised = cls._normalise_content(content).strip()
        if not normalised:
            return False
        try:
            payload = json.loads(normalised)
        except (TypeError, json.JSONDecodeError):
            return False

        def is_tool_call(value: Any) -> bool:
            return isinstance(value, dict) and (
                value.get("type") == "tool_call"
                or ("args" in value and "name" in value)
            )

        if is_tool_call(payload):
            return True
        return (
            isinstance(payload, list)
            and bool(payload)
            and all(is_tool_call(item) for item in payload)
        )

    @staticmethod
    def _replace_content(message: AIMessage, content: str) -> AIMessage:
        """Return the same assistant message with sanitized content."""
        return message.model_copy(update={"content": content})

    @hook_config(can_jump_to=["model", "end"])
    def after_model(
        self, state: FinalAnswerGuardState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Sanitize malformed model output and retry it once."""
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None
        if not isinstance(last_message, AIMessage):
            return None

        has_tool_calls = bool(getattr(last_message, "tool_calls", None))
        normalised = self._normalise_content(last_message.content)
        has_residue = self._is_tool_call_residue(last_message.content)
        if not has_residue and normalised.strip():
            return None

        if has_tool_calls:
            if has_residue:
                return {"messages": [self._replace_content(last_message, "")]}
            return None

        if not state.get("final_answer_guard_retried", False):
            return {
                "messages": [self._replace_content(last_message, "")],
                "final_answer_guard_retried": True,
                "jump_to": "model",
            }

        return {
            "messages": [
                self._replace_content(
                    last_message,
                    "I couldn't produce a readable answer. Please resend your question.",
                )
            ],
            "jump_to": "end",
        }
