from types import SimpleNamespace

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from src.middleware.docs_language_filter_middleware import (
    DocsLanguageFilterMiddleware,
)
from src.utils.doc_language import infer_doc_language


def _request(name, messages, args=None):
    return ToolCallRequest(
        tool_call={"id": "call-1", "name": name, "args": args or {}},
        tool=None,
        state={"messages": messages},
        runtime=SimpleNamespace(),
    )


def _search_result():
    return ToolMessage(
        content=(
            "Title: Python page\n"
            "Link: https://docs.langchain.com/oss/python/page\n\n"
            "Title: JavaScript page\n"
            "Link: https://docs.langchain.com/oss/javascript/page\n\n"
            "Title: Shared page\n"
            "Link: https://docs.langchain.com/langsmith/shared"
        ),
        tool_call_id="call-1",
    )


def test_infer_doc_language_from_python_turn_and_prior_turn():
    assert infer_doc_language([HumanMessage(content="Use PostgresSaver")]) == "python"
    assert (
        infer_doc_language(
            [
                HumanMessage(content="I use Python"),
                HumanMessage(content="How do I stream?"),
            ]
        )
        == "python"
    )


def test_infer_doc_language_from_javascript_and_ambiguous_turns():
    assert infer_doc_language([HumanMessage(content="How does useStream work?")]) == (
        "javascript"
    )
    assert infer_doc_language([HumanMessage(content="Show the TypeScript API")]) == (
        "javascript"
    )
    assert (
        infer_doc_language(
            [HumanMessage(content="Compare Python and JavaScript implementations")]
        )
        is None
    )


def test_filters_wrong_language_and_keeps_python_and_neutral_blocks():
    middleware = DocsLanguageFilterMiddleware()
    request = _request(
        "search_docs_by_lang_chain",
        [HumanMessage(content="I need the Python middleware docs")],
    )

    result = middleware.wrap_tool_call(request, lambda _: _search_result())

    assert "oss/python/page" in result.content
    assert "oss/javascript/page" not in result.content
    assert "oss/langsmith" not in result.content
    assert "langsmith/shared" in result.content
    assert "withheld 1" in result.content


def test_passes_through_when_all_search_blocks_are_wrong_language():
    middleware = DocsLanguageFilterMiddleware()
    content = (
        "Title: JavaScript page\nLink: https://docs.langchain.com/oss/javascript/page"
    )
    request = _request(
        "search_docs_by_lang_chain",
        [HumanMessage(content="Python docs")],
    )

    result = middleware.wrap_tool_call(
        request,
        lambda _: ToolMessage(content=content, tool_call_id="call-1"),
    )

    assert result.content == content


def test_rejects_wrong_language_filesystem_read():
    middleware = DocsLanguageFilterMiddleware()
    request = _request(
        "query_docs_filesystem_docs_by_lang_chain",
        [HumanMessage(content="Show me the Python docs")],
        {"command": "cat /docs/oss/javascript/agents.mdx"},
    )

    called = False

    def handler(_):
        nonlocal called
        called = True
        return ToolMessage(content="page body", tool_call_id="call-1")

    result = middleware.wrap_tool_call(request, handler)

    assert "Withheld filesystem read" in result.content
    assert "JavaScript" in result.content
    assert not called


def test_disabled_filter_leaves_tool_result_unchanged(monkeypatch):
    monkeypatch.setenv("DOCS_LANGUAGE_FILTER_ENABLED", "false")
    middleware = DocsLanguageFilterMiddleware()
    request = _request(
        "search_docs_by_lang_chain",
        [HumanMessage(content="Python docs")],
    )
    result = middleware.wrap_tool_call(request, lambda _: _search_result())

    assert "oss/javascript/page" in result.content
