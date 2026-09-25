import json
from types import SimpleNamespace

from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.runtime import ExecutionInfo, Runtime

from src.middleware.read_after_search_middleware import ReadAfterSearchMiddleware
from src.tools import pylon_tools


def _article(index: int, title: str) -> dict:
    return {
        "id": f"article-{index}",
        "title": title,
        "identifier": f"id-{index}",
        "slug": f"article-{index}",
        "collection_id": "collection-1",
        "is_published": True,
        "visibility_config": {"visibility": "public"},
    }


def _runtime() -> Runtime:
    return Runtime(
        execution_info=ExecutionInfo("checkpoint", "", "task", run_id="run-1")
    )


def test_support_search_requires_query_and_filters_titles(monkeypatch):
    monkeypatch.setattr(
        pylon_tools,
        "_fetch_all_articles",
        lambda: [_article(1, "Deployment timeout")],
    )
    monkeypatch.setattr(
        pylon_tools,
        "_fetch_collections",
        lambda: {"Troubleshooting": "collection-1"},
    )

    assert pylon_tools.search_support_articles.args_schema.model_fields[
        "query"
    ].is_required()
    result = json.loads(
        pylon_tools.search_support_articles.invoke({"query": "TIMEOUT"})
    )

    assert result["total"] == 1
    assert result["omitted"] == 0
    assert result["articles"][0]["title"] == "Deployment timeout"
    assert set(result["articles"][0]) == {"id", "title", "url", "collection"}


def test_support_search_is_bounded_and_never_returns_catalog(monkeypatch):
    articles = [_article(index, f"Timeout issue {index}") for index in range(25)]
    articles.append(_article(99, "Unrelated article"))
    monkeypatch.setattr(pylon_tools, "_fetch_all_articles", lambda: articles)
    monkeypatch.setattr(
        pylon_tools,
        "_fetch_collections",
        lambda: {"Troubleshooting": "collection-1"},
    )

    result = json.loads(
        pylon_tools.search_support_articles.invoke({"query": "timeout"})
    )

    assert result["total"] == 25
    assert result["omitted"] == 5
    assert len(result["articles"]) == 20
    assert (
        pylon_tools.search_support_articles.args_schema.model_fields[
            "collections"
        ].default
        is None
    )


def test_read_after_search_nudges_once_with_candidates():
    middleware = ReadAfterSearchMiddleware()
    runtime = _runtime()
    search_request = SimpleNamespace(
        tool_call={"name": "search_support_articles"}, runtime=runtime
    )
    middleware.wrap_tool_call(
        search_request,
        lambda _: ToolMessage(
            content=json.dumps({"articles": [{"id": "article-1"}]}),
            tool_call_id="call-1",
        ),
    )

    calls = []

    def handler(request):
        calls.append(request)
        return SimpleNamespace(result=[AIMessage(content="final answer")])

    request = ModelRequest(model=SimpleNamespace(), messages=[], runtime=runtime)
    response = middleware.wrap_model_call(request, handler)

    assert len(calls) == 2
    assert "article-1" in calls[1].messages[-1].content
    assert response.result[-1].content == "final answer"


def test_read_after_search_does_not_nudge_after_read():
    middleware = ReadAfterSearchMiddleware()
    runtime = _runtime()
    middleware.wrap_tool_call(
        SimpleNamespace(
            tool_call={"name": "search_docs_by_lang_chain"}, runtime=runtime
        ),
        lambda _: ToolMessage(
            content="Page: /docs/middleware.mdx", tool_call_id="call-1"
        ),
    )
    middleware.wrap_tool_call(
        SimpleNamespace(
            tool_call={"name": "query_docs_filesystem_docs_by_lang_chain"},
            runtime=runtime,
        ),
        lambda _: ToolMessage(content="full page content", tool_call_id="call-2"),
    )

    calls = []
    response = middleware.wrap_model_call(
        ModelRequest(model=SimpleNamespace(), messages=[], runtime=runtime),
        lambda request: (
            calls.append(request)
            or SimpleNamespace(result=[AIMessage(content="final answer")])
        ),
    )

    assert len(calls) == 1
    assert response.result[-1].content == "final answer"


def test_prompt_requires_read_grounding_for_enumerations():
    from src.prompts.docs_agent_prompt import docs_agent_prompt

    assert "For full-list or enumeration requests" in docs_agent_prompt
    assert "read-tool content" in docs_agent_prompt
    assert "DoclingChunker" not in docs_agent_prompt


def test_read_after_search_nudges_once_for_each_unpaired_search():
    middleware = ReadAfterSearchMiddleware()
    runtime = _runtime()
    for tool_name, content in (
        ("search_docs_by_lang_chain", "Page: /docs/middleware.mdx"),
        (
            "search_support_articles",
            json.dumps({"articles": [{"id": "article-1"}]}),
        ),
    ):
        middleware.wrap_tool_call(
            SimpleNamespace(tool_call={"name": tool_name}, runtime=runtime),
            lambda _, content=content: ToolMessage(
                content=content, tool_call_id=tool_name
            ),
        )

    calls = []
    middleware.wrap_model_call(
        ModelRequest(model=SimpleNamespace(), messages=[], runtime=runtime),
        lambda request: (
            calls.append(request)
            or SimpleNamespace(result=[AIMessage(content="final answer")])
        ),
    )

    assert len(calls) == 3
    nudge_text = "\n".join(call.messages[-1].content for call in calls[1:])
    assert "query_docs_filesystem_docs_by_lang_chain" in nudge_text
    assert "get_support_article_content" in nudge_text
