"""Detects and recovers from truncated final assistant responses."""

import logging
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)

_CONTINUE_PROMPT = (
    "Your previous answer appears to have been cut off mid-output "
    "(unclosed code fence or shell argument). Please continue exactly "
    "where you stopped, without repeating any content, and make sure "
    "to close all code fences and quoted arguments."
)


def _content_text(message: AIMessage) -> str:
    """Return message content as a flat string, handling list-of-parts content."""
    content = message.content
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def is_truncated(text: str) -> bool:
    """Heuristic: True when text ends mid-code-fence or mid-shell-argument."""
    if not text:
        return False

    fence_count = text.count("```")
    fence_unmatched = fence_count % 2 == 1

    stripped = text.rstrip()
    if not stripped:
        return False

    # Continuation backslash followed by an open `-H "` (or similar) header line
    # is a strong shell-truncation signal even when fences are balanced.
    last_line = stripped.splitlines()[-1]
    open_quote = last_line.count('"') % 2 == 1
    ends_with_continuation = stripped.endswith("\\")

    if fence_unmatched:
        return True

    if open_quote and ("-H" in last_line or ends_with_continuation):
        return True

    return False


class EnsureCompletionMiddleware(AgentMiddleware[AgentState]):
    """Auto-continues a truncated final assistant response once per turn."""

    def __init__(self, max_continuations: int = 1) -> None:
        """Initialize with the maximum number of auto-continuation turns per thread."""
        super().__init__()
        self.max_continuations = max_continuations

    def _last_ai_message(self, state: AgentState) -> AIMessage | None:
        for message in reversed(state.get("messages", [])):
            if isinstance(message, AIMessage):
                return message
        return None

    def after_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Inject a continuation turn when the latest AI message looks truncated."""
        message = self._last_ai_message(state)
        if message is None:
            return None

        # Skip messages that still have pending tool calls — those aren't
        # final responses and will be followed by tool execution anyway.
        if getattr(message, "tool_calls", None):
            return None

        text = _content_text(message)
        if not is_truncated(text):
            return None

        already_continued = sum(
            1
            for m in state.get("messages", [])
            if isinstance(m, HumanMessage)
            and getattr(m, "additional_kwargs", {}).get("completion_guard")
        )
        if already_continued >= self.max_continuations:
            logger.warning(
                "Detected truncated response after %d continuation(s); "
                "leaving output as-is for the user to follow up.",
                already_continued,
            )
            return None

        logger.info("Detected truncated assistant response; injecting continuation turn")
        return {
            "messages": [
                HumanMessage(
                    content=_CONTINUE_PROMPT,
                    additional_kwargs={"completion_guard": True},
                )
            ]
        }


__all__ = ["EnsureCompletionMiddleware", "is_truncated"]
