"""Tests for deterministic final-response documentation protection."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from src.middleware.docs_output_middleware import NO_RELEVANT_DOCS, DocsOutputMiddleware
from src.tools.link_check_tools import LinkCheckResult, check_links

VALID_URL = "https://docs.langchain.com/oss/python/langchain/overview"
UNKNOWN_URL = "https://docs.langchain.com/oss/python/langchain/expression-language"


def _guard(content, *history):
    message = AIMessage(content=content, id="answer", response_metadata={"keep": True})
    update = DocsOutputMiddleware().after_model(
        {"messages": [*history, message]}, SimpleNamespace()
    )
    if update is None:
        return message
    result = update["messages"][0]
    assert result.id == message.id
    assert result.response_metadata == message.response_metadata
    assert message.content == content
    return result


@pytest.mark.parametrize(
    "host", ["python.langchain.com", "js.langchain.com", "PYTHON.LANGCHAIN.COM:443"]
)
def test_banned_markdown_links_become_display_text(host):
    content = f'Reference: [Python docs](https://{host}/docs/ "Docs"). Other prose.'
    assert _guard(content).content == "Reference: Python docs. Other prose."


def test_banned_markdown_link_with_parentheses_and_angle_brackets():
    content = "Reference: [Docs](<https://python.langchain.com/docs/path_(detail)>)."
    assert _guard(content).content == "Reference: Docs."


@pytest.mark.parametrize(
    "heading", ["Relevant docs:", "**Relevant docs:**", "## Relevant docs:"]
)
def test_only_validated_non_banned_docs_bullets_remain(heading):
    result = ToolMessage(
        name="check_links",
        tool_call_id="check",
        content=f"Invalid links:\n  - {UNKNOWN_URL}: HTTP 404\n\nValid links:\n  - {VALID_URL}\n  - https://python.langchain.com/docs/",
    )
    content = (
        f"Unrelated prose.\n\n{heading}\n"
        f"- [Invalid]({UNKNOWN_URL})\n"
        "- [Unchecked](https://docs.langchain.com/unchecked)\n"
        "- [Banned](https://python.langchain.com/docs/)\n"
        f"- [Validated]({VALID_URL})\n"
    )
    assert _guard(content, result).content == (
        f"Unrelated prose.\n\n{heading}\n- [Validated]({VALID_URL})\n"
    )


@pytest.mark.parametrize("source", ["input", "invalid", "other_tool", "human", "error"])
def test_non_validation_sources_do_not_authorize_links(source):
    history = {
        "input": AIMessage(
            content="",
            tool_calls=[
                {"name": "check_links", "args": {"urls": [UNKNOWN_URL]}, "id": "check"}
            ],
        ),
        "invalid": ToolMessage(
            name="check_links",
            tool_call_id="check",
            content=f"Invalid links:\n  - {UNKNOWN_URL}: HTTP 404",
        ),
        "other_tool": ToolMessage(
            name="search_docs_by_lang_chain",
            tool_call_id="search",
            content=f"Valid links:\n  - {UNKNOWN_URL}",
        ),
        "human": HumanMessage(content=f"Valid links:\n  - {UNKNOWN_URL}"),
        "error": ToolMessage(
            name="check_links",
            tool_call_id="check",
            content=f"Valid links:\n  - {UNKNOWN_URL}",
            status="error",
        ),
    }[source]
    answer = _guard(f"Answer.\n\n**Relevant docs:**\n- [Docs]({UNKNOWN_URL})", history)
    assert UNKNOWN_URL not in answer.content


def test_unnamed_tool_result_resolves_check_links_by_call_id():
    call = AIMessage(
        content="",
        tool_calls=[
            {"name": "check_links", "args": {"urls": [VALID_URL]}, "id": "check"}
        ],
    )
    result = ToolMessage(tool_call_id="check", content=f"Valid links:\n  - {VALID_URL}")
    content = f"**Relevant docs:**\n- [Docs]({VALID_URL})"
    assert _guard(content, call, result).content == content


def test_valid_section_does_not_include_following_invalid_section():
    result = ToolMessage(
        name="check_links",
        tool_call_id="check",
        content=f"Valid links:\n  - {VALID_URL}\nInvalid links:\n  - {UNKNOWN_URL}: HTTP 404",
    )
    content = f"**Relevant docs:**\n- [Docs]({VALID_URL})\n- [Invalid]({UNKNOWN_URL})"
    answer = _guard(content, result)
    assert VALID_URL in answer.content
    assert UNKNOWN_URL not in answer.content


@pytest.mark.parametrize(
    "message",
    [
        AIMessage(
            content="Use [Docs](https://python.langchain.com/docs/)",
            tool_calls=[{"name": "search", "args": {}, "id": "search"}],
        ),
        HumanMessage(content="https://python.langchain.com/docs/"),
        ToolMessage(content="https://python.langchain.com/docs/", tool_call_id="tool"),
        AIMessage(
            content="Unrelated prose about moving your database and using old checkpoints."
        ),
        AIMessage(content="The documentation explains how to load old checkpoints."),
        AIMessage(content="Read the migration guide for older versions."),
        AIMessage(content="Read the docs first. The database moved to another server."),
        AIMessage(
            content="Documentation describes how migrated checkpoint data is loaded."
        ),
        AIMessage(content="Avoid legacy documentation sites."),
        AIMessage(content="Do not use legacy sites."),
    ],
)
def test_non_final_messages_and_unrelated_prose_are_unchanged(message):
    assert (
        DocsOutputMiddleware().after_model({"messages": [message]}, SimpleNamespace())
        is None
    )


def test_empty_state_is_unchanged():
    assert (
        DocsOutputMiddleware().after_model({"messages": []}, SimpleNamespace()) is None
    )


def test_unrelated_links_and_bullets_outside_docs_section_are_unchanged():
    content = "Unrelated prose with [a link](https://example.com/page).\n\n- Another [link](https://example.com/other)\n"
    assert _guard(content).content == content


def test_text_blocks_preserve_non_text_content_and_metadata():
    image = {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}
    content = [
        {
            "type": "text",
            "text": "Reference: [Docs](https://js.langchain.com/docs/)",
            "extra": "keep",
        },
        image,
        "\n**Relevant docs:**\n",
        {"type": "text", "text": f"- [Unchecked]({UNKNOWN_URL})\n"},
    ]
    answer = _guard(content)
    assert answer.content[0] == {
        "type": "text",
        "text": "Reference: Docs",
        "extra": "keep",
    }
    assert answer.content[1] == image
    assert answer.content[-1]["text"] == ""


def test_docs_section_continues_across_text_blocks():
    result = ToolMessage(
        name="check_links",
        tool_call_id="check",
        content=f"Valid links:\n  - {VALID_URL}",
    )
    content = [
        {"type": "text", "text": "**Relevant docs:**\n"},
        {
            "type": "text",
            "text": f"- [Docs]({VALID_URL})\n- [Unchecked]({UNKNOWN_URL})",
        },
    ]
    answer = _guard(content, result)
    assert answer.content[0] == content[0]
    assert answer.content[1]["text"] == f"- [Docs]({VALID_URL})\n"


@pytest.mark.parametrize(
    "claim",
    [
        "The expression-language documentation moved to legacy sites.",
        "The LCEL docs migrated to python.langchain.com.",
        "LCEL documentation still lives on https://python.langchain.com/docs/expression_language/.",
        "The current docs do not cover expression language.",
        "The documentation has been relocated to another site.",
        "The documentation remains on old Python reference sites.",
        "You should use the legacy sites instead.",
        "The **documentation** has **moved** to another site.",
        "Use the older documentation at [Python](https://python.langchain.com/docs/) and [JavaScript](https://js.langchain.com/docs/).",
    ],
)
def test_unsupported_migration_absence_and_legacy_recommendations_are_replaced(claim):
    answer = _guard(f"Unrelated prose. {claim} More unrelated prose.")
    assert (
        answer.content == f"Unrelated prose. {NO_RELEVANT_DOCS} More unrelated prose."
    )


def test_read_evidence_can_support_a_non_legacy_migration_claim():
    claim = "The documentation moved to the new official docs site."
    read = ToolMessage(
        name="query_docs_filesystem_docs_by_lang_chain",
        tool_call_id="read",
        content=claim,
    )
    assert _guard(claim, read).content == claim
    search = ToolMessage(
        name="search_docs_by_lang_chain", tool_call_id="search", content=claim
    )
    assert _guard(claim, search).content.strip() == NO_RELEVANT_DOCS


def test_empty_search_is_protected_end_to_end():
    @tool
    def search_docs_by_lang_chain(query: str) -> str:
        """Search for relevant documentation."""
        return "No results found."

    class ToolCallingModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    model = ToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_docs_by_lang_chain",
                        "args": {"query": "expression language"},
                        "id": "search",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "check_links",
                        "args": {"urls": [VALID_URL]},
                        "id": "check",
                    }
                ],
            ),
            AIMessage(
                content=(
                    "The LCEL documentation moved to legacy sites. "
                    "Use the old documentation at [Python LCEL](https://python.langchain.com/docs/expression_language/) "
                    "and [JavaScript LCEL](https://js.langchain.com/docs/expression_language/).\n\n"
                    "**Relevant docs:**\n"
                    "- [Python LCEL](https://python.langchain.com/docs/expression_language/)\n"
                    f"- [Unchecked LCEL]({UNKNOWN_URL})\n"
                    f"- [Overview]({VALID_URL})\n"
                )
            ),
        ]
    )
    graph = create_agent(
        model=model,
        tools=[search_docs_by_lang_chain, check_links],
        middleware=[DocsOutputMiddleware()],
    )
    with patch(
        "src.tools.link_check_tools._check_urls_async",
        new=AsyncMock(return_value=[LinkCheckResult(url=VALID_URL, valid=True)]),
    ):
        result = asyncio.run(
            graph.ainvoke(
                {
                    "messages": [
                        HumanMessage(
                            content="Where is expression-language documentation?"
                        )
                    ]
                }
            )
        )
    answer = result["messages"][-1].content
    assert NO_RELEVANT_DOCS in answer
    assert "python.langchain.com" not in answer
    assert "js.langchain.com" not in answer
    assert UNKNOWN_URL not in answer
    assert "moved" not in answer
    assert "legacy" not in answer
    assert "Use the old" not in answer
    assert f"- [Overview]({VALID_URL})" in answer
    assert (
        len(
            [
                message
                for message in result["messages"]
                if isinstance(message, AIMessage) and not message.tool_calls
            ]
        )
        == 1
    )


def test_guard_is_registered_on_the_managed_agent():
    source = Path("agent.py").read_text()
    assert (
        "from src.middleware.docs_output_middleware import DocsOutputMiddleware"
        in source
    )
    assert "docs_agent_middleware = [\n    DocsOutputMiddleware()," in source
