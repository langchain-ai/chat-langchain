"""Tests for ranked Pylon support article search."""

import json
from unittest.mock import patch

from src.tools.pylon_tools import search_support_articles

COLLECTIONS = {
    "General": "collection-general",
    "Troubleshooting": "collection-troubleshooting",
}


def _article(article_id, title, body, collection_id="collection-general"):
    return {
        "id": article_id,
        "title": title,
        "identifier": f"identifier-{article_id}",
        "slug": f"slug-{article_id}",
        "collection_id": collection_id,
        "is_published": True,
        "visibility_config": {"visibility": "public"},
        "current_published_content_html": body,
    }


def _search(articles, query, collections="all"):
    with patch("src.tools.pylon_tools._fetch_all_articles", return_value=articles):
        with patch(
            "src.tools.pylon_tools._fetch_collections", return_value=COLLECTIONS
        ):
            return json.loads(
                search_support_articles.invoke(
                    {"query": query, "collections": collections}
                )
            )


def test_title_matches_rank_above_body_matches():
    result = _search(
        [
            _article("body", "Troubleshooting guide", "This explains deployment."),
            _article("title", "Deployment troubleshooting", "General guidance."),
        ],
        "deployment troubleshooting",
    )

    assert result["articles"][0]["id"] == "title"
    assert result["articles"][0]["snippet"] == "General guidance."
    assert result["total_matched"] == 2
    assert result["returned"] == 2


def test_body_only_match_returns_matching_snippet():
    result = _search(
        [
            _article(
                "body",
                "How to configure an agent",
                "Use the tracing callback setting here.",
            )
        ],
        "callback",
    )

    assert result["articles"][0]["id"] == "body"
    assert "tracing callback setting" in result["articles"][0]["snippet"]
    assert len(result["articles"][0]["snippet"]) <= 300


def test_zero_match_returns_neutral_note():
    result = _search(
        [_article("one", "Deployment guide", "Configure deployments here.")],
        "unrelated billing topic",
    )

    assert result["total_matched"] == 0
    assert result["returned"] == 0
    assert result["articles"] == []
    assert result["note"] == (
        "No matching support articles found; the knowledge base responded normally."
    )


def test_collection_filter_combines_with_query():
    result = _search(
        [
            _article("general", "Deployment guide", "General deployment help."),
            _article(
                "troubleshooting",
                "Deployment errors",
                "Troubleshooting deployment failures.",
                "collection-troubleshooting",
            ),
        ],
        "deployment",
        "Troubleshooting",
    )

    assert result["total_matched"] == 1
    assert result["articles"][0]["id"] == "troubleshooting"
    assert result["articles"][0]["collection"] == "Troubleshooting"


def test_search_caps_results_at_ten():
    articles = [
        _article(str(index), f"Deployment article {index}", "Deployment details.")
        for index in range(12)
    ]

    result = _search(articles, "deployment")

    assert result["total_matched"] == 12
    assert result["returned"] == 10
    assert len(result["articles"]) == 10
    assert all(
        set(article) == {"id", "article_id", "title", "url", "collection", "snippet"}
        for article in result["articles"]
    )
