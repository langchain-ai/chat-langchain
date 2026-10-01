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

PRIVATE_ARTICLE = {
    **ARTICLE,
    "id": "private-uuid",
    "title": "Security Runbook",
    "current_published_content_html": "<p>Private content</p>",
    "visibility_config": {"visibility": "private"},
}

UNPUBLISHED_ARTICLE = {
    **ARTICLE,
    "id": "unpublished-uuid",
    "title": "Unpublished Login Guide",
    "current_published_content_html": "<p>Unpublished content</p>",
    "is_published": False,
}


@pytest.mark.parametrize(
    "article_id",
    ["uuid-123", "12345", "troubleshooting-login", "12345-troubleshooting-login"],
)
def test_get_support_article_content_resolves_supported_identifiers(article_id):
    with (
        patch("src.tools.pylon_tools._fetch_all_articles", return_value=[ARTICLE]),
        patch(
            "src.tools.pylon_tools._fetch_collections",
            return_value={"OSS (LangChain and LangGraph)": "oss-id"},
        ),
    ):
        result = get_support_article_content.invoke({"article_id": article_id})

    assert "ID: uuid-123" in result
    assert "Helpful content" in result


def test_get_support_article_content_suggests_closest_articles_on_miss():
    other_article = {**ARTICLE, "id": "uuid-456", "title": "Resetting Login"}
    with (
        patch(
            "src.tools.pylon_tools._fetch_all_articles",
            return_value=[ARTICLE, other_article],
        ),
        patch(
            "src.tools.pylon_tools._fetch_collections",
            return_value={"OSS (LangChain and LangGraph)": "oss-id"},
        ),
    ):
        result = get_support_article_content.invoke({"article_id": "login-help"})

    assert "Article ID login-help not found in knowledge base" in result
    assert "Troubleshooting Login (ID: uuid-123)" in result
    assert "Resetting Login (ID: uuid-456)" in result


@pytest.mark.parametrize(
    "article, forbidden_content",
    [
        (PRIVATE_ARTICLE, "Private content"),
        (UNPUBLISHED_ARTICLE, "Unpublished content"),
    ],
)
def test_get_support_article_content_rejects_non_public_articles(
    article, forbidden_content
):
    with (
        patch(
            "src.tools.pylon_tools._fetch_all_articles",
            return_value=[ARTICLE, article],
        ),
        patch(
            "src.tools.pylon_tools._fetch_collections",
            return_value={"OSS (LangChain and LangGraph)": "oss-id"},
        ),
    ):
        result = get_support_article_content.invoke({"article_id": article["id"]})

    assert f"Article ID {article['id']} not found in knowledge base" in result
    assert forbidden_content not in result


def test_get_support_article_content_suggestions_exclude_non_public_articles():
    with (
        patch(
            "src.tools.pylon_tools._fetch_all_articles",
            return_value=[PRIVATE_ARTICLE, UNPUBLISHED_ARTICLE, ARTICLE],
        ),
        patch(
            "src.tools.pylon_tools._fetch_collections",
            return_value={"OSS (LangChain and LangGraph)": "oss-id"},
        ),
    ):
        result = get_support_article_content.invoke({"article_id": "security-runbo"})

    assert "Security Runbook" not in result
    assert "private-uuid" not in result
    assert "Unpublished Login Guide" not in result
    assert "unpublished-uuid" not in result


def test_search_support_articles_normalizes_parenthetical_collection_words():
    with (
        patch("src.tools.pylon_tools._fetch_all_articles", return_value=[ARTICLE]),
        patch(
            "src.tools.pylon_tools._fetch_collections",
            return_value={"OSS (LangChain and LangGraph)": "oss-id"},
        ),
    ):
        result = search_support_articles.invoke(
            {"query": "login", "collections": " OSS (LangGraph and LangChain) "}
        )

    payload = json.loads(result)
    assert payload["total_matched"] == 1
    assert payload["articles"][0]["id"] == "uuid-123"
    assert payload["articles"][0]["article_id"] == "uuid-123"


def test_search_support_articles_reports_no_unmatched_collections_when_all_match():
    with (
        patch("src.tools.pylon_tools._fetch_all_articles", return_value=[ARTICLE]),
        patch(
            "src.tools.pylon_tools._fetch_collections",
            return_value={"OSS (LangChain and LangGraph)": "oss-id"},
        ),
    ):
        result = search_support_articles.invoke(
            {"query": "login", "collections": "OSS (LangChain and LangGraph)"}
        )

    payload = json.loads(result)
    assert payload["unmatched_collections"] == []


def test_search_support_articles_ignores_unmatched_collection_names():
    with (
        patch("src.tools.pylon_tools._fetch_all_articles", return_value=[ARTICLE]),
        patch(
            "src.tools.pylon_tools._fetch_collections",
            return_value={"OSS (LangChain and LangGraph)": "oss-id"},
        ),
    ):
        result = search_support_articles.invoke(
            {
                "query": "login",
                "collections": "Invented Collection, OSS (LangChain and LangGraph)",
            }
        )

    payload = json.loads(result)
    assert payload["total_matched"] == 1
    assert payload["unmatched_collections"] == ["Invented Collection"]


def test_search_support_articles_errors_when_all_collection_names_are_unmatched():
    with (
        patch("src.tools.pylon_tools._fetch_all_articles", return_value=[ARTICLE]),
        patch(
            "src.tools.pylon_tools._fetch_collections",
            return_value={"OSS (LangChain and LangGraph)": "oss-id"},
        ),
    ):
        result = search_support_articles.invoke(
            {"query": "login", "collections": "Invented Collection"}
        )

    payload = json.loads(result)
    assert "error" in payload
    assert "Invented Collection" in payload["error"]
