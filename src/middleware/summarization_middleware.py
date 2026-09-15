"""Summarization middleware with shared model retry and fallback behavior."""

from typing import Any

from langchain.agents.middleware import AgentState, SummarizationMiddleware
from langchain_core.messages import AnyMessage, RemoveMessage, SystemMessage
from langchain_core.messages.utils import get_buffer_string
from langchain_core.runnables import Runnable
from langgraph.runtime import Runtime

from src.utils.message_utils import latest_user_message_index


class CustomSummarizationMiddleware(SummarizationMiddleware):
    """Use a custom runnable for summary generation."""

    def __init__(self, *args: Any, summary_model: Runnable, **kwargs: Any) -> None:
        """Initialize the middleware with a separate summary-generation runnable."""
        super().__init__(*args, **kwargs)
        self.summary_model = summary_model

    @staticmethod
    def _build_new_messages(summary: str) -> list[SystemMessage]:
        return [
            SystemMessage(
                content=f"Here is a summary of the conversation to date:\n\n{summary}",
                additional_kwargs={"lc_source": "summarization"},
            )
        ]

    def _reconstruct_messages(
        self, messages: list[AnyMessage], summary: str, cutoff_index: int
    ) -> dict[str, list[AnyMessage]]:
        cutoff_index = self._clamp_cutoff_index(messages, cutoff_index)
        _, preserved_messages = self._partition_messages(messages, cutoff_index)
        new_messages = self._build_new_messages(summary)
        return {
            "messages": [
                RemoveMessage(id="__remove_all__"),
                *new_messages,
                *preserved_messages,
            ]
        }

    @staticmethod
    def _clamp_cutoff_index(messages: list[AnyMessage], cutoff_index: int) -> int:
        latest_user_index = latest_user_message_index(messages)
        return (
            min(cutoff_index, latest_user_index)
            if latest_user_index >= 0
            else cutoff_index
        )

    def before_model(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Summarize history while preserving the current user turn."""
        messages = state["messages"]
        self._ensure_message_ids(messages)
        total_tokens = self.token_counter(messages)
        if not self._should_summarize(messages, total_tokens):
            return None
        cutoff_index = self._clamp_cutoff_index(
            messages, self._determine_cutoff_index(messages)
        )
        if cutoff_index <= 0:
            return None
        summary = self._create_summary(
            self._partition_messages(messages, cutoff_index)[0]
        )
        return self._reconstruct_messages(messages, summary, cutoff_index)

    async def abefore_model(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Asynchronously summarize history while preserving the current user turn."""
        messages = state["messages"]
        self._ensure_message_ids(messages)
        total_tokens = self.token_counter(messages)
        if not self._should_summarize(messages, total_tokens):
            return None
        cutoff_index = self._clamp_cutoff_index(
            messages, self._determine_cutoff_index(messages)
        )
        if cutoff_index <= 0:
            return None
        summary = await self._acreate_summary(
            self._partition_messages(messages, cutoff_index)[0]
        )
        return self._reconstruct_messages(messages, summary, cutoff_index)

    def _create_summary(self, messages_to_summarize: list[AnyMessage]) -> str:
        """Generate a summary using the configured retry/fallback summary model."""
        if not messages_to_summarize:
            return "No previous conversation history."

        trimmed_messages = self._trim_messages_for_summary(messages_to_summarize)
        if not trimmed_messages:
            return "Previous conversation was too long to summarize."

        formatted_messages = get_buffer_string(trimmed_messages)

        try:
            response = self.summary_model.invoke(
                self.summary_prompt.format(messages=formatted_messages).rstrip(),
                config={"metadata": {"lc_source": "summarization"}},
            )
            return response.text.strip()
        except Exception as e:
            return f"Error generating summary: {e!s}"

    async def _acreate_summary(self, messages_to_summarize: list[AnyMessage]) -> str:
        """Generate a summary using the configured retry/fallback summary model."""
        if not messages_to_summarize:
            return "No previous conversation history."

        trimmed_messages = self._trim_messages_for_summary(messages_to_summarize)
        if not trimmed_messages:
            return "Previous conversation was too long to summarize."

        formatted_messages = get_buffer_string(trimmed_messages)

        try:
            response = await self.summary_model.ainvoke(
                self.summary_prompt.format(messages=formatted_messages).rstrip(),
                config={"metadata": {"lc_source": "summarization"}},
            )
            return response.text.strip()
        except Exception as e:
            return f"Error generating summary: {e!s}"


__all__ = ["CustomSummarizationMiddleware"]
