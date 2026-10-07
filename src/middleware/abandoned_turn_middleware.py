"""Close abandoned user turns before processing the latest request."""

from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime


class AbandonedTurnMiddleware(AgentMiddleware):
    """Mark unanswered earlier turns as cancelled and remove their tool pairs."""

    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Close abandoned turns without changing the newest human turn."""
        messages = state["messages"]
        human_indices = [
            index
            for index, message in enumerate(messages)
            if isinstance(message, HumanMessage)
        ]
        if len(human_indices) < 2:
            return None

        cleaned_messages = list(messages[: human_indices[0]])
        changed = False
        for start, end in zip(human_indices[:-1], human_indices[1:], strict=True):
            human = messages[start]
            segment = messages[start + 1 : end]
            marker_id = f"abandoned-{human.id}"
            if any(
                isinstance(message, AIMessage)
                and (
                    message.id == marker_id
                    or (message.text.strip() and not message.tool_calls)
                )
                for message in segment
            ):
                cleaned_messages.extend(messages[start:end])
                continue

            changed = True
            tool_call_ids = {
                tool_call["id"]
                for message in segment
                if isinstance(message, AIMessage)
                for tool_call in message.tool_calls
            }
            cleaned_messages.extend(
                [
                    human,
                    AIMessage(
                        content="(This earlier request was cancelled before an answer was produced.)",
                        id=marker_id,
                    ),
                ]
            )
            cleaned_messages.extend(
                message
                for message in segment
                if not (
                    isinstance(message, AIMessage)
                    and message.tool_calls
                    or isinstance(message, ToolMessage)
                    and message.tool_call_id in tool_call_ids
                )
            )

        if not changed:
            return None

        cleaned_messages.extend(messages[human_indices[-1] :])
        # add_messages appends new IDs; reset to insert markers beside their humans.
        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *cleaned_messages]}
