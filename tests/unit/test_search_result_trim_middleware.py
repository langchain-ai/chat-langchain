"""Tests for bounded documentation search results."""

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from src.middleware.search_result_trim_middleware import SearchResultTrimMiddleware


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
    message, tool_name="search_docs_by_lang_chain", asynchronous=False
):
    request = SimpleNamespace(tool_call={"name": tool_name})
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
    )


@pytest.mark.parametrize("leading_whitespace", [False, True])
def test_large_whitespace_content_is_also_bounded(leading_whitespace):
    body = " " * 100000 + "short" if leading_whitespace else "short" + " " * 100000
    content = f"Title: Result\nLink: link\nPage: /page\nContent: {body}"
    result = invoke_middleware(ToolMessage(content=content, tool_call_id="search-1"))
    snippet = "..." if leading_whitespace else "short..."
    assert (
        result.content == f"Title: Result\nLink: link\nPage: /page\nContent: {snippet}"
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
