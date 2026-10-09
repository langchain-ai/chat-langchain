"""Tests for bounded documentation search results."""

import asyncio

import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from src.middleware.duplicate_call_guard_middleware import DuplicateCallGuardMiddleware
from src.middleware.search_result_trim_middleware import SearchResultTrimMiddleware


def _hit(index, content=None):
    if content is None:
        content = f"Content for hit {index}.\n" * 200
    return (
        f"Title: Hit {index}\n"
        f"Link: https://docs.langchain.com/hit-{index}\n"
        f"Page: /hit-{index}\n"
        f"Content: {content}"
    )


def _invoke(result, asynchronous, name="search_docs_by_lang_chain"):
    request = ToolCallRequest(
        tool_call={"name": name, "id": "call-1", "args": {}},
        tool=None,
        state={"messages": []},
        runtime=None,
    )
    middleware = SearchResultTrimMiddleware()

    def handler(received):
        assert received is request
        return result

    async def async_handler(received):
        return handler(received)

    if asynchronous:
        return asyncio.run(middleware.awrap_tool_call(request, async_handler))
    return middleware.wrap_tool_call(request, handler)


def _message(content, status="success"):
    return ToolMessage(
        content=content,
        name="search_docs_by_lang_chain",
        tool_call_id="call-1",
        status=status,
    )


def _assert_hits(text):
    assert text.count("Title:") == 6
    for index in range(6):
        snippet = (f"Content for hit {index}.\n" * 200)[:200] + "..."
        assert _hit(index, snippet) in text
    assert "Title: Hit 6" not in text


@pytest.mark.parametrize("asynchronous", [False, True])
def test_string_results_are_bounded(asynchronous):
    original = _message("\n\n".join(_hit(index) for index in range(11)))
    result = _invoke(original, asynchronous)

    _assert_hits(result.content)
    assert original.content.count("Title:") == 11
    assert result.tool_call_id == original.tool_call_id
    assert result.name == original.name
    assert result.status == original.status


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("separate_blocks", [False, True])
@pytest.mark.parametrize("status", ["success", "error"])
def test_text_blocks_share_hit_budget_and_preserve_metadata(
    asynchronous, separate_blocks, status
):
    texts = [_hit(index) for index in range(11)]
    if not separate_blocks:
        texts = ["\n\n".join(texts)]
    original = _message(
        [{"type": "text", "text": text, "id": "block-id"} for text in texts],
        status=status,
    )
    result = _invoke(original, asynchronous)

    _assert_hits("\n\n".join(block["text"] for block in result.content))
    assert all(block["id"] == "block-id" for block in result.content)
    assert result.tool_call_id == original.tool_call_id
    assert result.name == original.name
    assert result.status == original.status


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "name", ["query_docs_filesystem_docs_by_lang_chain", "search_support_articles"]
)
@pytest.mark.parametrize("text_blocks", [False, True])
def test_other_tools_pass_through_unchanged(asynchronous, name, text_blocks):
    content = "\n\n".join(_hit(index) for index in range(11))
    if text_blocks:
        content = [{"type": "text", "text": content}]
    original = ToolMessage(content=content, name=name, tool_call_id="call-1")

    assert _invoke(original, asynchronous, name) is original


@pytest.mark.parametrize("asynchronous", [False, True])
def test_short_snippets_and_unrecognized_results_are_unchanged(asynchronous):
    for text in [_hit(0, "Short content"), "No results found."]:
        assert _invoke(_message(text), asynchronous).content == text
    command = Command(update={"messages": []})
    assert _invoke(command, asynchronous) is command


@pytest.mark.parametrize("asynchronous", [False, True])
def test_non_text_blocks_are_preserved(asynchronous):
    block = {"type": "image", "url": "https://docs.langchain.com/image.png"}
    blocks = [block, {"type": "text", "text": ""}]
    result = _invoke(_message(blocks), asynchronous)

    assert result.content == blocks


def test_duplicate_guard_caches_trimmed_search_results():
    guard = DuplicateCallGuardMiddleware()
    trimmer = SearchResultTrimMiddleware()
    original = _message("\n\n".join(_hit(index) for index in range(11)))
    calls = []

    async def handler(request):
        calls.append(request)
        return original

    async def trimmed_handler(request):
        return await trimmer.awrap_tool_call(request, handler)

    async def invoke():
        results = []
        for call_id in ["call-1", "call-2"]:
            request = ToolCallRequest(
                tool_call={
                    "name": "search_docs_by_lang_chain",
                    "id": call_id,
                    "args": {"query": "middleware"},
                },
                tool=None,
                state={"messages": [HumanMessage(content="Question")]},
                runtime=None,
            )
            results.append(await guard.awrap_tool_call(request, trimmed_handler))
        return results

    first, second = asyncio.run(invoke())

    assert len(calls) == 1
    _assert_hits(first.content)
    _assert_hits(second.content)
    assert "already made on this turn" in second.content
    assert second.tool_call_id == "call-2"
