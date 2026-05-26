"""Summarization middleware with shared model retry and fallback behavior."""

from typing import Any

from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import AnyMessage
from langchain_core.messages.utils import get_buffer_string
from langchain_core.runnables import Runnable


class CustomSummarizationMiddleware(SummarizationMiddleware):
    """Use a custom runnable for summary generation."""

    def __init__(self, *args: Any, summary_model: Runnable, **kwargs: Any) -> None:
        """Initialize the middleware with a separate summary-generation runnable."""
        super().__init__(*args, **kwargs)
        self.summary_model = summary_model

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
