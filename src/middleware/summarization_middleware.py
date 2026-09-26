"""Summarization middleware with shared model retry and fallback behavior."""

from typing import Any

from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import get_buffer_string
from langchain_core.runnables import Runnable


class CustomSummarizationMiddleware(SummarizationMiddleware):
    """Use a custom runnable for summary generation."""

    def __init__(self, *args: Any, summary_model: Runnable, **kwargs: Any) -> None:
        """Initialize the middleware with a separate summary-generation runnable."""
        super().__init__(*args, **kwargs)
        self.summary_model = summary_model

    @staticmethod
    def _trim_stale_tool_results(messages: list[AnyMessage]) -> list[AnyMessage]:
        """Replace tool results from turns before the latest user message."""
        latest_human_index = next(
            (
                index
                for index in range(len(messages) - 1, -1, -1)
                if isinstance(messages[index], HumanMessage)
            ),
            len(messages),
        )
        trimmed_messages = []
        for index, message in enumerate(messages):
            if index < latest_human_index and isinstance(message, ToolMessage):
                content_length = (
                    len(message.content)
                    if isinstance(message.content, str)
                    else len(str(message.content))
                )
                tool_name = message.name or "unknown"
                trimmed_messages.append(
                    message.model_copy(
                        update={
                            "content": f"[stale tool result elided: {tool_name}, {content_length} chars]"
                        }
                    )
                )
            else:
                trimmed_messages.append(message)
        return trimmed_messages

    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Trim stale tool results before applying summarization."""
        messages = state["messages"]
        trimmed_messages = self._trim_stale_tool_results(messages)
        result = super().before_model({**state, "messages": trimmed_messages}, runtime)
        if result is None and trimmed_messages != messages:
            return {"messages": trimmed_messages}
        return result

    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Trim stale tool results before applying summarization asynchronously."""
        messages = state["messages"]
        trimmed_messages = self._trim_stale_tool_results(messages)
        result = await super().abefore_model(
            {**state, "messages": trimmed_messages}, runtime
        )
        if result is None and trimmed_messages != messages:
            return {"messages": trimmed_messages}
        return result

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
