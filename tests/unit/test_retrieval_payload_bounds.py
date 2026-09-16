"""Tests for bounded support article and MCP retrieval payloads."""

import json

from connectors.mcp import MCP_DOCS_CHARACTER_BUDGET, _truncate_content
from src.tools import pylon_tools


def _article(article_id: str, title: str, collection_id: str = "collection-1") -> dict:
    return {
        "id": article_id,
        "title": title,
        "is_published": True,
        "visibility_config": {"visibility": "public"},
        "identifier": article_id,
        "slug": title.lower().replace(" ", "-"),
        "collection_id": collection_id,
    }


def test_search_support_articles_enforces_limit_and_reports_total(monkeypatch):
    articles = [_article(f"a-{index}", f"Article {index}") for index in range(30)]
    monkeypatch.setattr(pylon_tools, "_fetch_all_articles", lambda: articles)
    monkeypatch.setattr(pylon_tools, "_fetch_collections", lambda: {"General": "collection-1"})

    result = json.loads(
        pylon_tools.search_support_articles.invoke({"limit": 100, "collections": "all"})
    )

    assert result["total"] == 30
    assert len(result["articles"]) == pylon_tools.MAX_SUPPORT_ARTICLE_RESULTS
    assert result["truncated"] is True
    assert "url" not in result["articles"][0]
    assert set(result["articles"][0]) == {"id", "title", "collection"}


def test_search_support_articles_filters_titles_before_limit(monkeypatch):
    articles = [
        _article("a-1", "Billing setup"),
        _article("a-2", "Authentication guide"),
        _article("a-3", "Billing troubleshooting"),
    ]
    monkeypatch.setattr(pylon_tools, "_fetch_all_articles", lambda: articles)
    monkeypatch.setattr(pylon_tools, "_fetch_collections", lambda: {"General": "collection-1"})

    result = json.loads(
        pylon_tools.search_support_articles.invoke(
            {"query": "BILLING", "limit": 1, "collections": "all"}
        )
    )

    assert result["total"] == 2
    assert [article["title"] for article in result["articles"]] == ["Billing setup"]
    assert result["truncated"] is True


def test_mcp_truncates_oversized_text_block():
    bounded, truncated = _truncate_content("line\n" * 10_000)

    assert truncated is True
    assert len(bounded) <= MCP_DOCS_CHARACTER_BUDGET
    assert bounded.endswith("]")
    assert "narrow the query" in bounded


def test_mcp_preserves_small_content():
    content = [{"type": "text", "text": "small result"}]

    bounded, truncated = _truncate_content(content)

    assert bounded == content
    assert truncated is False
