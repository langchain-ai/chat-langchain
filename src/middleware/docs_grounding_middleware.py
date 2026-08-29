"""Enforce research and link validation for product answers."""

from __future__ import annotations

import re
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.runtime import Runtime
from typing_extensions import NotRequired

LINK_TOOL_NAME = "check_links"
URL_PATTERN = re.compile(r"https?://\S+")
PRICING_PATTERN = re.compile(
    r"\b(pric(?:e|ing)|plan(?:s)?|billing|quota|cost|trace limits?|seats?|pay[- ]as[- ]you[- ]go)\b",
    re.IGNORECASE,
)
GREETING_PATTERN = re.compile(
    r"^(?:hi|hello|hey|good morning|good afternoon|good evening)[!. ,]*$",
    re.IGNORECASE,
)
IDENTITY_PATTERN = re.compile(
    r"\b(?:who are you|what can you do|how can you help|what do you help with)\b",
    re.IGNORECASE,
)
DECLINE_PATTERN = re.compile(
    r"\b(?:outside (?:my|the assistant's) scope|cannot help with|can't help with|not able to help with)\b",
    re.IGNORECASE,
)
CLARIFICATION_PATTERN = re.compile(
    r"\b(?:could you|can you|please)\b.*\?|\bwhich\b.*\?\s*$|\bwhat\b.*\?\s*$",
    re.IGNORECASE | re.DOTALL,
)


class DocsGroundingState(AgentState, total=False):
    """State fields used by the grounding guard."""

    grounding_guard_attempts: NotRequired[int]


class DocsGroundingMiddleware(AgentMiddleware[DocsGroundingState]):
    """Require content retrieval and validated links before product answers."""

    state_schema = DocsGroundingState

    def after_model(
        self, state: DocsGroundingState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Prevent an ungrounded final response from ending the turn."""
        messages = state.get("messages", [])
        latest_ai = self._latest_ai_message(messages)
        if latest_ai is None or latest_ai.tool_calls:
            return None

        response_text = self._message_text(latest_ai.content)
        if not response_text or self._is_permitted_tool_free_reply(
            messages, response_text
        ):
            return None

        turn_messages = self._current_turn_messages(messages)
        if not self._has_relevant_content(turn_messages) or not self._links_are_valid(
            turn_messages, response_text
        ):
            attempts = state.get("grounding_guard_attempts", 0) + 1
            return {"grounding_guard_attempts": attempts, "jump_to": "model"}

        return {"grounding_guard_attempts": 0}

    def _latest_ai_message(self, messages: list[Any]) -> AIMessage | None:
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                return message
        return None

    def _current_turn_messages(self, messages: list[Any]) -> list[Any]:
        for index in range(len(messages) - 1, -1, -1):
            if getattr(messages[index], "type", None) == "human":
                return messages[index:]
        return messages

    def _message_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and isinstance(block.get("text"), str)
            )
        return str(content)

    def _is_permitted_tool_free_reply(self, messages: list[Any], response: str) -> bool:
        user_text = self._message_text(
            next(
                (
                    message.content
                    for message in reversed(messages)
                    if getattr(message, "type", None) == "human"
                ),
                "",
            )
        ).strip()
        normalized = response.strip()
        return bool(
            GREETING_PATTERN.fullmatch(user_text)
            or IDENTITY_PATTERN.search(user_text)
            or DECLINE_PATTERN.search(normalized)
            or (normalized.endswith("?") and CLARIFICATION_PATTERN.search(normalized))
        )

    def _has_relevant_content(self, messages: list[Any]) -> bool:
        pricing_question = any(
            getattr(message, "type", None) == "human"
            and PRICING_PATTERN.search(self._message_text(message.content))
            for message in messages
        )
        allowed_tools = (
            {"fetch_langchain_pricing"}
            if pricing_question
            else {
                "query_docs_filesystem_docs_by_lang_chain",
                "get_support_article_content",
            }
        )
        return any(
            isinstance(message, ToolMessage)
            and message.name in allowed_tools
            and self._successful_tool_content(message.content)
            for message in messages
        )

    def _successful_tool_content(self, content: Any) -> bool:
        text = self._message_text(content).strip().lower()
        return bool(text) and not text.startswith(("error:", "no results", '{"error'))

    def _links_are_valid(self, messages: list[Any], response: str) -> bool:
        urls = [url.rstrip(".,;:") for url in URL_PATTERN.findall(response)]
        if not urls:
            return True
        valid_results = [
            self._message_text(message.content)
            for message in messages
            if isinstance(message, ToolMessage) and message.name == LINK_TOOL_NAME
        ]
        if not valid_results:
            return False
        valid_text = "\n".join(valid_results)
        return all(
            re.search(rf"^\s*-\s+{re.escape(url)}(?:\s|$)", valid_text, re.MULTILINE)
            and "Valid links:" in valid_text
            for url in urls
        )


__all__ = ["DocsGroundingMiddleware", "DocsGroundingState"]
