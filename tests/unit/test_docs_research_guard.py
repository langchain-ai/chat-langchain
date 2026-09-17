import asyncio
import os
from unittest.mock import MagicMock

os.environ.setdefault("GOOGLE_API_KEY", "test")
os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ.setdefault("ANTHROPIC_API_KEY", "test")

import pytest
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from src.agent.config import DEFAULT_MODEL, FALLBACK_MODELS
from src.middleware.docs_research_guard_middleware import DocsResearchGuardMiddleware


@tool
def search_docs_by_lang_chain(query: str) -> str:
    """Search LangChain documentation."""
    return query


@pytest.mark.parametrize(
    "model_id",
    [DEFAULT_MODEL.id, *(model.id for model in FALLBACK_MODELS)],
)
def test_research_tool_choice_is_accepted_by_configured_provider(model_id: str):
    model = init_chat_model(model=model_id)

    model.bind_tools(
        [search_docs_by_lang_chain],
        tool_choice="search_docs_by_lang_chain",
    )


def test_research_guard_uses_bare_tool_name():
    middleware = DocsResearchGuardMiddleware()
    request = ModelRequest(
        model=MagicMock(),
        messages=[HumanMessage(content="How do I add middleware?")],
    )
    seen_choices = []

    async def handler(next_request):
        seen_choices.append(next_request.tool_choice)
        return ModelResponse(result=[AIMessage(content="I need to search.")])

    asyncio.run(middleware.awrap_model_call(request, handler))

    assert seen_choices == [None, "search_docs_by_lang_chain"]
