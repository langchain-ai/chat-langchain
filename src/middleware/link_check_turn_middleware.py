"""Reset link validation state at the start of each agent turn."""

from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.runtime import Runtime

from src.tools.link_check_tools import reset_link_check_turn


class LinkCheckTurnMiddleware(AgentMiddleware):
    """Reset URLs validated during the current agent turn."""

    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict | None:
        """Reset the per-turn link validation registry."""
        reset_link_check_turn()
        return None
