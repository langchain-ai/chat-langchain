"""Tests for support knowledge-base failure disclosures."""

import asyncio

from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.middleware.source_disclosure_middleware import SourceDisclosureMiddleware

_DISCLOSURE = (
    "Support articles could not be consulted, so this answer is based on official "
    "documentation only."
)


def _request(*tool_messages: ToolMessage) -> ModelRequest:
    return ModelRequest(
        model=object(),
        messages=[HumanMessage(content="How do I configure retention?") , *tool_messages],
    )


def _run(middleware: SourceDisclosureMiddleware, request: ModelRequest, message: AIMessage):
    async def handler(request: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[message])

    return asyncio.run(middleware.awrap_model_call(request, handler)).result[0]


def test_errored_search_produces_one_disclosure():
    message = _run(
        SourceDisclosureMiddleware(),
        _request(
            ToolMessage(
                content="Pylon API returned HTTP 401",
                name="search_support_articles",
                status="error",
                tool_call_id="search",
            )
        ),
        AIMessage(content="**Answer**\n\nRetention is configured in the account settings.\n\n**Relevant docs:**\n- [Guide](https://docs.example.com/retention)"),
    )

    assert message.content.count(_DISCLOSURE) == 1
    assert message.content.splitlines()[2] == _DISCLOSURE


def test_article_content_leaves_answer_unchanged():
    original = "**Answer**\n\nRetention is configured in the account settings.\n\n**Relevant docs:**\n- [Guide](https://docs.example.com/retention)"
    message = _run(
        SourceDisclosureMiddleware(),
        _request(
            ToolMessage(
                content="Article title: Retention\nRetention settings are available to admins.",
                name="get_support_article_content",
                tool_call_id="content",
            )
        ),
        AIMessage(content=original),
    )

    assert message.content == original


def test_empty_support_results_produce_disclosure():
    message = _run(
        SourceDisclosureMiddleware(),
        _request(
            ToolMessage(
                content="No articles returned from API",
                name="search_support_articles",
                tool_call_id="search",
            ),
            ToolMessage(
                content="No content available.",
                name="get_support_article_content",
                tool_call_id="content",
            ),
        ),
        AIMessage(content="**Answer**\n\nThe policy applies to all workspaces.\n\n**Relevant docs:**"),
    )

    assert message.content.count(_DISCLOSURE) == 1


def test_existing_disclosure_is_not_duplicated():
    original = f"**Answer**\n\n{_DISCLOSURE}\n\nThe policy applies to all workspaces.\n\n**Relevant docs:**"
    message = _run(
        SourceDisclosureMiddleware(),
        _request(
            ToolMessage(
                content="PylonUnavailableError: unavailable",
                name="search_support_articles",
                tool_call_id="search",
            )
        ),
        AIMessage(content=original),
    )

    assert message.content == original


def test_turn_without_support_call_is_unchanged():
    original = "**Answer**\n\nThe policy applies to all workspaces.\n\n**Relevant docs:**"
    message = _run(
        SourceDisclosureMiddleware(),
        _request(
            ToolMessage(
                content="Official docs result",
                name="search_docs_by_lang_chain",
                tool_call_id="docs",
            )
        ),
        AIMessage(content=original),
    )

    assert message.content == original


def test_block_content_preserves_representation_and_ordering():
    message = _run(
        SourceDisclosureMiddleware(),
        _request(
            ToolMessage(
                content="Article ID 123 not found in knowledge base.",
                name="get_support_article_content",
                tool_call_id="content",
            )
        ),
        AIMessage(
            content=[
                {"type": "text", "text": "**Answer**\n\nThe policy applies."},
                {"type": "text", "text": "\n\n**Relevant docs:**\n- [Guide](https://docs.example.com/policy)"},
            ]
        ),
    )

    assert isinstance(message.content, list)
    text = "\n".join(block["text"] for block in message.content)
    assert text.count(_DISCLOSURE) == 1
    assert text.index("**Answer**") < text.index(_DISCLOSURE) < text.index("**Relevant docs:**")
    assert text.endswith("**Relevant docs:**\n- [Guide](https://docs.example.com/policy)")
