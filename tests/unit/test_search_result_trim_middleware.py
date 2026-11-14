"""Tests for bounded documentation search results."""

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.types import Command

from src.middleware.search_result_trim_middleware import (
    TRUNCATION_NOTICE,
    SearchResultTrimMiddleware,
)


@pytest.fixture
def search_payload():
    return "\n\n".join(
        f"Title: Result {index}\n"
        f"Link: https://docs.langchain.com/page-{index}\n"
        f"Page: /page-{index}\n"
        f"Content: {'documentation ' * 2000}"
        for index in range(10)
    )


def invoke_middleware(
    message, tool_name="search_docs_by_lang_chain", asynchronous=False, query=""
):
    request = SimpleNamespace(tool_call={"name": tool_name, "args": {"query": query}})
    middleware = SearchResultTrimMiddleware()
    calls = []

    def handler(received):
        calls.append(received)
        return message

    async def async_handler(received):
        return handler(received)

    if asynchronous:
        result = asyncio.run(middleware.awrap_tool_call(request, async_handler))
    else:
        result = middleware.wrap_tool_call(request, handler)
    assert calls == [request]
    return result


def assert_trimmed(text):
    assert len(text) < 5000
    text = text.removesuffix("\n" + TRUNCATION_NOTICE)
    sections = text.split("Title: ")[1:]
    assert len(sections) == 10
    for index, section in enumerate(sections):
        assert section.startswith(f"Result {index}\n")
        assert f"Link: https://docs.langchain.com/page-{index}\n" in section
        assert f"Page: /page-{index}\n" in section
        snippet = section.split("Content: ", 1)[1].rstrip()
        assert snippet.endswith("...")
        assert len(snippet[:-3]) <= 300


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("text_blocks", [False, True])
def test_ten_results_preserve_metadata_with_bounded_snippets(
    search_payload, asynchronous, text_blocks
):
    content = (
        [{"type": "text", "text": search_payload, "extra": "keep"}]
        if text_blocks
        else search_payload
    )
    message = ToolMessage(
        content=content,
        tool_call_id="search-1",
        name="search_docs_by_lang_chain",
        id="message-1",
        artifact={"source": "docs"},
    )
    result = invoke_middleware(message, asynchronous=asynchronous)
    text = result.content[0]["text"] if text_blocks else result.content
    assert_trimmed(text)
    assert message.content == content
    assert result.model_dump(exclude={"content"}) == message.model_dump(
        exclude={"content"}
    )
    if text_blocks:
        assert result.content[0]["extra"] == "keep"


@pytest.mark.parametrize("asynchronous", [False, True])
def test_non_target_tool_is_untouched(search_payload, asynchronous):
    message = ToolMessage(content=search_payload, tool_call_id="support-1")
    assert (
        invoke_middleware(message, "search_support_articles", asynchronous) is message
    )


@pytest.mark.parametrize(
    "content",
    [
        "No results found.",
        "Content: a response without a result title",
        "Title: Missing content\nLink: https://docs.langchain.com\nPage: /docs",
        "Title: Malformed content\nContent without colon\nLink: link\nPage: /page",
        "Title: Empty\nContent:\nLink: link\nPage: /page",
        "Title: Short\nContent: short snippet\nLink: link\nPage: /page",
    ],
)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_missing_malformed_and_short_content_is_unchanged(content, asynchronous):
    message = ToolMessage(content=content, tool_call_id="search-1")
    assert invoke_middleware(message, asynchronous=asynchronous).content == content


def test_metadata_after_content_and_following_results_survives():
    text = (
        f"Title: First\nContent: {'x' * 2000}\nLink: first\nPage: /first\n"
        "Title: Second\nLink: second\nPage: /second\nContent: short"
    )
    result = invoke_middleware(ToolMessage(content=text, tool_call_id="search-1"))
    assert result.content == (
        f"Title: First\nContent: {'x' * 300}...\nLink: first\nPage: /first\n"
        "Title: Second\nLink: second\nPage: /second\nContent: short"
        "\n" + TRUNCATION_NOTICE
    )


@pytest.mark.parametrize("leading_whitespace", [False, True])
def test_large_whitespace_content_is_also_bounded(leading_whitespace):
    body = " " * 100000 + "short" if leading_whitespace else "short" + " " * 100000
    content = f"Title: Result\nLink: link\nPage: /page\nContent: {body}"
    result = invoke_middleware(ToolMessage(content=content, tool_call_id="search-1"))
    snippet = "..." if leading_whitespace else "short..."
    assert (
        result.content
        == f"Title: Result\nLink: link\nPage: /page\nContent: {snippet}\n{TRUNCATION_NOTICE}"
    )


def test_non_text_and_malformed_text_blocks_are_unchanged():
    content = [
        {"type": "image_url", "image_url": {"url": "https://example.com/image"}},
        {"type": "text"},
        {"type": "text", "text": None},
        "plain content",
    ]
    result = invoke_middleware(ToolMessage(content=content, tool_call_id="search-1"))
    assert result.content == content


@pytest.mark.parametrize("asynchronous", [False, True])
def test_command_result_is_unchanged(asynchronous):
    command = Command(update={"messages": []})
    assert invoke_middleware(command, asynchronous=asynchronous) is command


@pytest.mark.parametrize("asynchronous", [False, True])
def test_trims_mcp_prefixed_docs_search(search_payload, asynchronous):
    message = ToolMessage(content=search_payload, tool_call_id="call-1")
    result = invoke_middleware(
        message,
        tool_name="langchainDocs_search_docs_by_lang_chain",
        asynchronous=asynchronous,
    )
    assert len(result.content) < 5000
    assert result.content.count("...") == 10


def test_ignores_tools_that_only_contain_the_name(search_payload):
    message = ToolMessage(content=search_payload, tool_call_id="call-1")
    result = invoke_middleware(message, tool_name="search_docs_by_lang_chainx")
    assert result.content == message.content


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "query", ["langchain_tavily python package", "LANGCHAIN-TAVILY"]
)
def test_query_preserves_late_table_row(query, asynchronous):
    row = "| langchain-tavily | Official Tavily integration |"
    body = "| Provider | Package |\n" + "| --- | --- |\n" * 30 + row
    result = invoke_middleware(
        ToolMessage(content=f"Title: Integrations\nContent: {body}", tool_call_id="s"),
        query=query,
        asynchronous=asynchronous,
    )
    assert row in result.content
    assert result.content.endswith(TRUNCATION_NOTICE)


@pytest.mark.parametrize("query", ["", "how the python langchain package", "unmatched"])
def test_no_matching_lines_preserve_original_snippet(query):
    body = "x" * 320 + "\nother information"
    result = invoke_middleware(
        ToolMessage(content=f"Title: Docs\nContent: {body}", tool_call_id="s"),
        query=query,
    )
    assert (
        result.content == f"Title: Docs\nContent: {'x' * 300}...\n{TRUNCATION_NOTICE}"
    )


def test_matching_sections_are_capped_and_each_text_block_is_marked():
    body = "x" * 320 + "\n" + "tavily " * 1000
    content = [{"type": "text", "text": f"Title: Docs\nContent: {body}"}] * 2
    result = invoke_middleware(
        ToolMessage(content=content, tool_call_id="s"), query="tavily"
    )
    for block in result.content:
        snippet, notice = (
            block["text"].split("\n", 1)[1].removeprefix("Content: ").split("\n", 1)
        )
        assert len(snippet) == 1500
        assert notice == TRUNCATION_NOTICE


@pytest.mark.parametrize(
    "reader",
    [
        "read_integration_page",
        "query_docs_filesystem_docs_by_lang_chain",
        "langchainDocs_query_docs_filesystem_docs_by_lang_chain",
    ],
)
@pytest.mark.parametrize("status", ["success", "error"])
def test_only_successful_page_reads_satisfy_current_turn(reader, status):
    middleware = SearchResultTrimMiddleware()
    messages = [
        HumanMessage(content="Does the package exist?"),
        ToolMessage(
            content="Search", name="search_docs_by_lang_chain", tool_call_id="s"
        ),
        ToolMessage(content="Page", name=reader, tool_call_id="r", status=status),
        AIMessage(content="It exists", id="answer"),
    ]
    result = middleware.after_model({"messages": messages}, None)
    assert (result is None) == (status == "success")


@pytest.mark.parametrize(
    "old_message",
    [
        ToolMessage(content="Old page", name="read_integration_page", tool_call_id="r"),
        SystemMessage(content="Read a page", name="docs_page_read_reminder"),
    ],
)
def test_previous_turn_read_or_reminder_does_not_skip_current_turn(old_message):
    messages = [
        old_message,
        HumanMessage(content="New question"),
        ToolMessage(
            content="Search", name="search_docs_by_lang_chain", tool_call_id="s"
        ),
        AIMessage(content="No package exists", id="answer"),
    ]
    assert (
        SearchResultTrimMiddleware().after_model({"messages": messages}, None)[
            "jump_to"
        ]
        == "model"
    )


@pytest.mark.parametrize("searched", [False, True])
def test_tool_call_response_is_not_redirected(searched):
    messages = [HumanMessage(content="Question")]
    if searched:
        messages.append(
            ToolMessage(
                content="Search", name="search_docs_by_lang_chain", tool_call_id="s"
            )
        )
    messages.append(
        AIMessage(
            content="",
            tool_calls=[{"name": "read_integration_page", "args": {}, "id": "r"}],
        )
    )
    assert (
        SearchResultTrimMiddleware().after_model({"messages": messages}, None) is None
    )


def test_query_fact_later_on_first_line_is_retained():
    result = invoke_middleware(
        ToolMessage(
            content="Title: Docs\nContent: " + "x" * 320 + " tavily", tool_call_id="s"
        ),
        query="tavily",
    )
    assert "tavily" in result.content
