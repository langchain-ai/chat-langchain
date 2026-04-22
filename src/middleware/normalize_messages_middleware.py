# Normalize inbound messages so ChatAnthropic doesn't crash on docs-page-context traffic.
#
# The frontend prepends a SystemMessage ("Context about the user's current page...")
# to every turn. Our agent's own system_prompt is also prepended by create_agent.
# ChatAnthropic._format_messages rejects multiple non-consecutive system messages
# and raises ValueError, killing the run and returning an empty response.
#
# This middleware runs before any other middleware and rewrites any inbound
# SystemMessage on the state into a HumanMessage, preserving the content. The
# rewritten message carries the original message id so the ``add_messages``
# reducer on ``AgentState`` replaces the existing SystemMessage in place rather
# than appending a duplicate.
import logging

from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)


class NormalizeInboundSystemMessagesMiddleware(AgentMiddleware):
    """Convert any inbound SystemMessage on the conversation into a HumanMessage.

    Prevents ChatAnthropic from raising
    ``ValueError('Received multiple non-consecutive system messages.')`` when
    the frontend injects docs-page context as a SystemMessage alongside the
    agent's own system prompt. Content is preserved verbatim; only the message
    type changes so the downstream model adapter sees exactly one system
    message (the one injected by ``create_agent``).
    """

    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict | None:
        messages = state.get("messages") or []
        if not any(isinstance(m, SystemMessage) for m in messages):
            return None

        rewritten: list = [RemoveMessage(id=REMOVE_ALL_MESSAGES)]
        converted = 0
        for m in messages:
            if isinstance(m, SystemMessage):
                rewritten.append(HumanMessage(content=m.content))
                converted += 1
            else:
                rewritten.append(m)

        logger.info(
            "NormalizeInboundSystemMessagesMiddleware: converted "
            f"{converted} inbound SystemMessage(s) to HumanMessage"
        )
        return {"messages": rewritten}
