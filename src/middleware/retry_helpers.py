"""Helpers for constructing provider-compatible retry requests."""

from __future__ import annotations

from collections.abc import Sequence

from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage


def build_repair_request(
    request: ModelRequest,
    response_messages: Sequence[BaseMessage],
    repair_instruction: str,
    system_message: SystemMessage,
) -> ModelRequest:
    """Build a retry request whose final turn is a human message."""
    draft = "\n\n".join(_message_text(message) for message in response_messages)
    content = f"{repair_instruction}\n\nRejected assistant draft:\n{draft}"
    return request.override(
        messages=[*request.messages, HumanMessage(content=content)],
        system_message=system_message,
    )


def _message_text(message: BaseMessage) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)
