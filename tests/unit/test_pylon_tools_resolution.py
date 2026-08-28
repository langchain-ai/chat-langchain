"""Tests for tolerant Pylon article and collection resolution."""

import json
from unittest.mock import patch

from src.tools import pylon_tools

ARTICLE = {
    "id": "uuid-123",
    "identifier": "4567",
    "slug": "reset-api-key",
    "title": "Reset API key",
    "is_published": True,
    "visibility_config": {"visibility": "public"},
    "collection_id": "collection-oss",
    "current_published_content_html": "<p>Article content</p>",
}

COLLECTIONS = {
    "OSS (LangChain and LangGraph)": "collection-oss",
    "General": "collection-general",
}


def _article_result(article_id):
    with (
        patch.object(pylon_tools, "_fetch_all_articles", return_value=[ARTICLE.copy()]),
        patch.object(pylon_tools, "_fetch_collections", return_value=COLLECTIONS),
    ):
        return pylon_tools.get_support_article_content.invoke({"article_id": article_id})


def test_article_resolves_by_uuid():
    assert "Article content" in _article_result("uuid-123")


def test_article_resolves_by_numeric_identifier():
    assert "Article content" in _article_result("4567")


def test_article_resolves_by_identifier_slug():
    assert "Article content" in _article_result("4567-reset-api-key")


def test_article_resolves_by_support_url():
    result = _article_result(
        "https://support.langchain.com/articles/4567-reset-api-key"
    )
    assert "Article content" in result


def test_unresolved_article_returns_actionable_hint():
    result = _article_result("missing")
    assert "Use an article UUID, numeric identifier, identifier-slug" in result


def test_collection_word_order_variant_resolves():
    articles = [ARTICLE.copy()]
    with (
        patch.object(pylon_tools, "_fetch_all_articles", return_value=articles),
        patch.object(pylon_tools, "_fetch_collections", return_value=COLLECTIONS),
    ):
        result = json.loads(
            pylon_tools.search_support_articles.invoke(
                {"collections": "OSS (LangGraph and LangChain)"}
            )
        )
    assert result["total"] == 1
    assert result["unmatched_collections"] == []


def test_all_inside_comma_separated_list_disables_filter():
    articles = [ARTICLE.copy()]
    with (
        patch.object(pylon_tools, "_fetch_all_articles", return_value=articles),
        patch.object(pylon_tools, "_fetch_collections", return_value=COLLECTIONS),
    ):
        result = json.loads(
            pylon_tools.search_support_articles.invoke(
                {"collections": "General, all"}
            )
        )
    assert result["total"] == 1
    assert result["unmatched_collections"] == []


def test_unknown_collection_mixed_with_valid_collection_succeeds():
    articles = [ARTICLE.copy()]
    with (
        patch.object(pylon_tools, "_fetch_all_articles", return_value=articles),
        patch.object(pylon_tools, "_fetch_collections", return_value=COLLECTIONS),
    ):
        result = json.loads(
            pylon_tools.search_support_articles.invoke(
                {"collections": "General, Unknown"}
            )
        )
    assert result["unmatched_collections"] == ["Unknown"]
    assert result["available_collections"] == list(COLLECTIONS)
