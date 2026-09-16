"""Tests for Pylon support article tools."""

import json
from unittest.mock import patch

import pytest

from src.tools.pylon_tools import get_support_article_content, search_support_articles

ARTICLE = {
    "id": "uuid-123",
    "identifier": "12345",
    "slug": "troubleshooting-login",
    "title": "Troubleshooting Login",
    "collection_id": "oss-id",
    "is_published": True,
    "visibility_config": {"visibility": "public"},
    "current_published_content_html": "<p>Helpful content</p>",
}


@pytest.mark.parametrize(
    "article_id",
    ["uuid-123", "12345", "12345-troubleshooting-login"],
)
def test_get_support_article_content_resolves_supported_identifiers(article_id):
    with patch(
        "src.tools.pylon_tools._fetch_all_articles", return_value=[ARTICLE]
    ), patch(
        "src.tools.pylon_tools._fetch_collections",
        return_value={"OSS (LangChain and LangGraph)": "oss-id"},
    ):
        result = get_support_article_content.invoke({"article_id": article_id})

    assert "ID: uuid-123" in result
    assert "Helpful content" in result


def test_get_support_article_content_suggests_closest_articles_on_miss():
    other_article = {**ARTICLE, "id": "uuid-456", "title": "Resetting Login"}
    with patch(
        "src.tools.pylon_tools._fetch_all_articles",
        return_value=[ARTICLE, other_article],
    ), patch(
        "src.tools.pylon_tools._fetch_collections",
        return_value={"OSS (LangChain and LangGraph)": "oss-id"},
    ):
        result = get_support_article_content.invoke({"article_id": "login-help"})

    assert "Article ID login-help not found in knowledge base" in result
    assert "Troubleshooting Login (ID: uuid-123)" in result
    assert "Resetting Login (ID: uuid-456)" in result


def test_search_support_articles_normalizes_parenthetical_collection_words():
    with patch(
        "src.tools.pylon_tools._fetch_all_articles", return_value=[ARTICLE]
    ), patch(
        "src.tools.pylon_tools._fetch_collections",
        return_value={"OSS (LangChain and LangGraph)": "oss-id"},
    ):
        result = search_support_articles.invoke(
            {"collections": " OSS (LangGraph and LangChain) "}
        )

    payload = json.loads(result)
    assert payload["total"] == 1
    assert payload["articles"][0]["id"] == "uuid-123"
    assert payload["articles"][0]["article_id"] == "uuid-123"
