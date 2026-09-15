"""Utilities for working with conversation messages."""

from langchain_core.messages import BaseMessage


def latest_user_message_index(messages: list[BaseMessage]) -> int:
    """Return the index of the latest non-summary human message."""
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if (
            getattr(message, "type", None) == "human"
            and getattr(message, "additional_kwargs", {}).get("lc_source")
            != "summarization"
        ):
            return index
    return -1
