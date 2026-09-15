import asyncio
from unittest.mock import MagicMock

from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.middleware.support_kb_disclosure_middleware import (
    DISCLOSURE,
    SupportKBDisclosureMiddleware,
)


def test_support_kb_error_is_disclosed_before_relevant_docs():
    request = ModelRequest(
        model=MagicMock(),
        messages=[
            HumanMessage(content="How does retention work?"),
            ToolMessage(
                content='{"error": "Knowledge base unavailable"}',
                name="search_support_articles",
                tool_call_id="call-1",
                status="error",
            ),
        ]
    )
    response = ModelResponse(
        result=[
            AIMessage(content="**Retention is configurable.**\n\nRelevant docs:\n- [Docs](https://docs.example.com)")
        ]
    )

    async def handler(_request):
        return response

    result = asyncio.run(
        SupportKBDisclosureMiddleware().awrap_model_call(request, handler)
    )

    content = result.result[0].content
    assert DISCLOSURE in content
    assert content.index(DISCLOSURE) < content.index("Relevant docs:")
