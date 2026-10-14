"""Test bounded support title search and its prompt contract."""

import ast
import json
import re
from pathlib import Path
from unittest.mock import patch

import pytest
import requests

from src.prompts.docs_agent_prompt import docs_agent_prompt
from src.tools import pylon_tools
from tests.unit.test_pylon_pagination import (
    ARTICLE_PAGE_1,
    ARTICLE_PAGE_2,
    ARTICLE_PAGE_3,
    _make_response,
)


@pytest.fixture
def articles(monkeypatch):
    published = [
        {
            **article,
            "is_published": True,
            "visibility_config": {"visibility": "public"},
            "identifier": article["id"],
            "slug": "support-guide",
            "collection_id": "deployment",
            "current_published_content_html": "<p>Support guidance</p>",
        }
        for article in ARTICLE_PAGE_1 + ARTICLE_PAGE_2 + ARTICLE_PAGE_3
    ]
    monkeypatch.setattr(pylon_tools, "_articles_cache", published)
    monkeypatch.setattr(
        pylon_tools,
        "_collections_cache",
        {"LangSmith Deployment": "deployment", "General": "general"},
    )
    return published


def test_case_insensitive_token_scores_and_stable_ties(articles):
    titles = [
        "Deployment guide",
        "DEPLOYMENT, tracing deployment!",
        "Tracing basics",
        "Deployment security",
        "Redeployment guide",
    ]
    for article, title in zip(articles, titles):
        article["title"] = title

    raw = pylon_tools.search_support_articles.invoke({"query": "dEpLoYmEnT TRACING"})
    result = json.loads(raw)

    assert set(result) == {"collections", "query", "total_matched", "articles"}
    assert result["query"] == "dEpLoYmEnT TRACING"
    assert result["total_matched"] == 4
    assert [article["id"] for article in result["articles"]] == ["a2", "a1", "a3", "a4"]
    assert all(
        set(article) == {"id", "title", "collection"} for article in result["articles"]
    )
    assert "url" not in raw.lower()
    assert raw == json.dumps(result, separators=(",", ":"))
    assert "\n" not in raw


@pytest.mark.parametrize(
    "limit, expected_count", [(1, 1), (2, 2), (15, 5), (0, 0), (-1, 0)]
)
def test_limit_and_total_before_limiting(articles, limit, expected_count):
    result = json.loads(
        pylon_tools.search_support_articles.invoke({"query": "article", "limit": limit})
    )
    assert len(result["articles"]) == expected_count
    assert result["total_matched"] == len(articles)


def test_all_collections_broad_query_is_filtered_and_bounded(articles):
    published = [
        {**articles[index % len(articles)], "id": f"article-{index}"}
        for index in range(225)
    ]
    published[-1]["title"] = "Unrelated title"
    with patch.object(pylon_tools, "_articles_cache", published):
        result = json.loads(
            pylon_tools.search_support_articles.invoke(
                {"query": "article", "collections": "all"}
            )
        )
    assert result["collections"] == "all"
    assert result["total_matched"] == 224
    assert len(result["articles"]) == 15
    assert [article["id"] for article in result["articles"]] == [
        f"article-{index}" for index in range(15)
    ]


def test_collection_filtering_before_matching(articles):
    articles[1]["collection_id"] = "general"
    result = json.loads(
        pylon_tools.search_support_articles.invoke(
            {"query": "article", "collections": " general "}
        )
    )
    assert result["total_matched"] == 1
    assert result["articles"] == [
        {"id": "a2", "title": "Article 2", "collection": "General"}
    ]
    result = json.loads(
        pylon_tools.search_support_articles.invoke(
            {"query": "article", "collections": "general,LangSmith Deployment"}
        )
    )
    assert result["total_matched"] == 5


@pytest.mark.parametrize("query", ["unrelated", "", "   ", "!?"])
def test_no_token_matches_returns_compact_empty_result(articles, query):
    raw = pylon_tools.search_support_articles.invoke({"query": query})
    result = {"collections": "all", "query": query, "total_matched": 0, "articles": []}
    assert raw == json.dumps(result, separators=(",", ":"))


@pytest.mark.parametrize(
    "invalid_fields",
    [
        {"is_published": False},
        {"visibility_config": {"visibility": "private"}},
        {"title": "Untitled"},
        {"title": ""},
        {"identifier": ""},
        {"slug": ""},
    ],
)
def test_existing_public_article_filter_is_preserved(articles, invalid_fields):
    articles[0].update(invalid_fields)
    result = json.loads(
        pylon_tools.search_support_articles.invoke({"query": "article"})
    )
    assert result["total_matched"] == 4
    assert "a1" not in [article["id"] for article in result["articles"]]


@pytest.mark.parametrize("empty_articles", [None, []])
def test_empty_api_response_is_unchanged(articles, empty_articles):
    with patch.object(pylon_tools, "_fetch_all_articles", return_value=empty_articles):
        raw = pylon_tools.search_support_articles.invoke({"query": "article"})
    assert raw == json.dumps(
        {
            "collections": "all",
            "total": 0,
            "articles": [],
            "note": "No articles returned from API",
        },
        indent=2,
    )


def test_no_published_articles_is_unchanged(articles):
    for article in articles:
        article["is_published"] = False
    assert pylon_tools.search_support_articles.invoke({"query": "article"}) == (
        "No published articles available in the knowledge base."
    )


def test_empty_collection_result_is_unchanged(articles):
    raw = pylon_tools.search_support_articles.invoke(
        {"query": "article", "collections": "General"}
    )
    assert raw == json.dumps(
        {
            "collections": "General",
            "total": 0,
            "articles": [],
            "note": "No articles found",
        },
        indent=2,
    )


def test_unknown_collection_error_is_unchanged(articles):
    raw = pylon_tools.search_support_articles.invoke(
        {"query": "article", "collections": "Missing"}
    )
    assert raw == json.dumps(
        {
            "error": "Collection 'Missing' not found. Available collections: LangSmith Deployment, General"
        },
        indent=2,
    )


def test_collection_fetch_error_is_unchanged(articles):
    with patch.object(
        pylon_tools, "_fetch_collections", side_effect=RuntimeError("failure")
    ):
        raw = pylon_tools.search_support_articles.invoke({"query": "article"})
    assert raw == json.dumps(
        {"error": "Failed to fetch collections: failure"}, indent=2
    )


@pytest.mark.parametrize(
    "error, message",
    [
        (
            ValueError("PYLON_API_KEY not configured in .env"),
            "PYLON_API_KEY not configured in .env",
        ),
        (requests.exceptions.ConnectionError("network failure"), "network failure"),
        (RuntimeError("failure"), "Unexpected error: failure"),
    ],
)
def test_article_fetch_errors_are_unchanged(articles, error, message):
    with patch.object(pylon_tools, "_fetch_all_articles", side_effect=error):
        raw = pylon_tools.search_support_articles.invoke({"query": "article"})
    assert raw == json.dumps({"error": message}, indent=2)


def test_prompt_query_to_paginated_search_and_article_read(articles, monkeypatch):
    instructions = (Path(__file__).resolve().parents[2] / "instructions.md").read_text()
    support_section = r"### 4\..*?(?=### 5\.)"
    assert (
        re.search(support_section, instructions, re.DOTALL).group()
        == re.search(support_section, docs_agent_prompt, re.DOTALL).group()
    )
    kb_guidance = r"   - \*\*For KB\*\*:.*"
    assert (
        re.search(kb_guidance, instructions).group()
        == re.search(kb_guidance, docs_agent_prompt).group()
    )
    schema = pylon_tools.search_support_articles.get_input_schema().model_json_schema()
    assert schema["required"] == ["query"]
    assert schema["properties"]["limit"]["default"] == 15
    example = re.search(
        r"\*\*Usage:\*\* `(search_support_articles\(query=.*?\))`", instructions
    )
    assert example is not None
    parsed_call = ast.parse(example.group(1), mode="eval").body
    arguments = {
        keyword.arg: ast.literal_eval(keyword.value) for keyword in parsed_call.keywords
    }
    articles[0]["title"] = "Deployment guide"
    monkeypatch.setattr(pylon_tools, "_articles_cache", None)
    monkeypatch.setenv("PYLON_KB_ID", "kb-123")
    monkeypatch.setenv("PYLON_API_KEY", "fake-key")

    with patch.object(pylon_tools.requests, "get") as mock_get:
        mock_get.side_effect = [
            _make_response(articles[:2], next_cursor="next-page"),
            _make_response(articles[2:]),
        ]
        result = json.loads(pylon_tools.search_support_articles.invoke(arguments))
        assert result["total_matched"] == 1
        assert result["articles"] == [
            {
                "id": "a1",
                "title": "Deployment guide",
                "collection": "LangSmith Deployment",
            }
        ]
        content = pylon_tools.get_support_article_content.invoke(
            {"article_id": result["articles"][0]["id"]}
        )
        assert "URL: https://support.langchain.com/articles/a1-support-guide" in content
        assert "<p>Support guidance</p>" in content
        assert mock_get.call_count == 2
