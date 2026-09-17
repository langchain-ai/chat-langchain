"""Ensure model turns that need research call the documentation search tool."""

from collections.abc import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware, ModelCallResult, ModelRequest
from langchain_core.messages import AIMessage, ToolMessage


class DocsResearchGuardMiddleware(AgentMiddleware):
    """Retry a model response without a docs search using a portable tool choice."""

    def __init__(self, research_tool_name: str = "search_docs_by_lang_chain") -> None:
        """Initialize the guard with the required research tool name."""
        super().__init__()
        self.research_tool_name = research_tool_name

    def _research_already_used(self, request: ModelRequest) -> bool:
        return any(
            isinstance(message, ToolMessage) and message.name == self.research_tool_name
            for message in request.messages
        )

    def _response_used_research(self, response: ModelCallResult) -> bool:
        return any(
            isinstance(message, AIMessage)
            and any(
                call.get("name") == self.research_tool_name
                for call in message.tool_calls
            )
            for message in response.result
        )

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelCallResult]],
    ) -> ModelCallResult:
        """Retry a response without research using a provider-portable choice."""
        response = await handler(request)
        if (
            request.tool_choice is None
            and not self._research_already_used(request)
            and not self._response_used_research(response)
        ):
            return await handler(request.override(tool_choice=self.research_tool_name))
        return response


__all__ = ["DocsResearchGuardMiddleware"]
