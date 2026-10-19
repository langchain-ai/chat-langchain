"""Test authorization failures through Pylon tools and retry middleware."""

import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import requests
from langchain_core.messages import ToolMessage

from src.middleware.tool_retry_middleware import ToolRetryMiddleware
from src.tools import pylon_tools as pylon

ARTICLE = {
    "id": "article-1",
    "title": "Support article",
    "identifier": "123",
    "slug": "support-article",
    "is_published": True,
    "visibility_config": {"visibility": "public"},
    "collection_id": "collection-1",
    "current_published_content_html": "Article content",
}


@pytest.fixture(autouse=True)
def reset_pylon(monkeypatch):
    monkeypatch.setattr(pylon, "_articles_cache", None)
    monkeypatch.setattr(pylon, "_collections_cache", None)
    monkeypatch.setenv("PYLON_API_KEY", "test-secret")
    monkeypatch.setenv("PYLON_KB_ID", "kb-123")


def response(status, data):
    result = requests.Response()
    result.status_code = status
    result._content = json.dumps({"data": data}).encode()
    result.url = "https://api.usepylon.com/knowledge-bases/kb-123/articles"
    return result


def invoke(tool_name):
    if tool_name == "search_support_articles":
        return pylon.search_support_articles.invoke({})
    return pylon.get_support_article_content.invoke({"article_id": "article-1"})


@pytest.mark.parametrize("status", [401, 403])
@pytest.mark.parametrize("fetch", [pylon._fetch_all_articles, pylon._fetch_collections])
def test_fetch_authorization_failure(monkeypatch, status, fetch):
    monkeypatch.setattr(pylon.requests, "get", Mock(return_value=response(status, [])))
    with pytest.raises(pylon.PylonUnavailableError) as error:
        fetch()
    assert str(status) in str(error.value)
    assert "Check or rotate PYLON_API_KEY" in str(error.value)
    assert "test-secret" not in str(error.value)


@pytest.mark.parametrize("status", [401, 403])
@pytest.mark.parametrize(
    "tool_name", ["search_support_articles", "get_support_article_content"]
)
@pytest.mark.parametrize("failure_source", ["articles", "collections"])
def test_tools_propagate_authorization_failure(
    monkeypatch, status, tool_name, failure_source
):
    if failure_source == "collections":
        monkeypatch.setattr(pylon, "_articles_cache", [ARTICLE])
    monkeypatch.setattr(pylon.requests, "get", Mock(return_value=response(status, [])))
    with pytest.raises(pylon.PylonUnavailableError):
        invoke(tool_name)


@pytest.mark.parametrize(
    "tool_name", ["search_support_articles", "get_support_article_content"]
)
@pytest.mark.parametrize("status", [401, 403])
@pytest.mark.asyncio
async def test_middleware_surfaces_authorization_failure(
    monkeypatch, caplog, tool_name, status
):
    monkeypatch.setattr(pylon.requests, "get", Mock(return_value=response(status, [])))
    request = SimpleNamespace(tool_call={"name": tool_name, "id": "call-1"})

    async def call_tool(request):
        return invoke(request.tool_call["name"])

    handler = AsyncMock(side_effect=call_tool)
    with caplog.at_level(logging.ERROR):
        result = await ToolRetryMiddleware(initial_delay=0).awrap_tool_call(
            request, handler
        )
    assert handler.await_count == 1
    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call-1"
    assert result.name == tool_name
    assert "Support knowledge base unavailable" in result.content
    assert "PYLON_API_KEY" not in result.content
    assert "test-secret" not in result.content + caplog.text
    assert any(record.levelno == logging.ERROR for record in caplog.records)
    assert "Check or rotate PYLON_API_KEY" in caplog.text


@pytest.mark.parametrize(
    "tool_name", ["search_support_articles", "get_support_article_content"]
)
def test_other_request_failures_keep_tool_output(monkeypatch, tool_name):
    monkeypatch.setattr(
        pylon.requests,
        "get",
        Mock(side_effect=requests.ConnectionError("connection lost")),
    )
    result = invoke(tool_name)
    if tool_name == "search_support_articles":
        assert json.loads(result) == {"error": "connection lost"}
    else:
        assert result == "Error fetching article: connection lost"


@pytest.mark.parametrize("fetch", [pylon._fetch_all_articles, pylon._fetch_collections])
def test_other_http_failures_remain_request_errors(monkeypatch, fetch):
    monkeypatch.setattr(pylon.requests, "get", Mock(return_value=response(503, [])))
    with pytest.raises(requests.HTTPError):
        fetch()


def test_successful_search_and_read(monkeypatch):
    monkeypatch.setattr(
        pylon.requests,
        "get",
        Mock(
            side_effect=[
                response(200, [ARTICLE]),
                response(
                    200,
                    [
                        {
                            "id": "collection-1",
                            "title": "General",
                            "visibility_config": {"visibility": "public"},
                        }
                    ],
                ),
            ]
        ),
    )
    result = json.loads(invoke("search_support_articles"))
    assert result["total"] == 1
    assert result["articles"][0]["collection"] == "General"
    assert "Article content" in invoke("get_support_article_content")


@pytest.mark.asyncio
async def test_transient_middleware_failure_retries():
    request = SimpleNamespace(tool_call={"name": "docs", "id": "call-1"})
    expected = ToolMessage(content="Docs", tool_call_id="call-1")
    handler = AsyncMock(
        side_effect=[requests.ConnectionError("connection error"), expected]
    )
    result = await ToolRetryMiddleware(initial_delay=0).awrap_tool_call(
        request, handler
    )
    assert result == expected
    assert handler.await_count == 2


@pytest.mark.asyncio
async def test_no_results_keeps_normalized_output():
    request = SimpleNamespace(tool_call={"name": "docs", "id": "call-1"})
    handler = AsyncMock(side_effect=ValueError("no results found"))
    result = await ToolRetryMiddleware(initial_delay=0).awrap_tool_call(
        request, handler
    )
    assert result.content == "No results found."
    assert handler.await_count == 1
