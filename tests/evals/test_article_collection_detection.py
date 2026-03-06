"""Tests for get_article_content collection detection.

Verifies that get_article_content uses the actual collection_id from article
data (via _fetch_collections) instead of keyword heuristics on the title.
"""

from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_article(article_id, title, collection_id, identifier="123", slug="test-slug"):
    return {
        "id": article_id,
        "title": title,
        "collection_id": collection_id,
        "identifier": identifier,
        "slug": slug,
        "current_published_content_html": "<p>Article body</p>",
        "is_published": True,
        "visibility_config": {"visibility": "public"},
    }


# Collection map returned by _fetch_collections(): {name: id}
MOCK_COLLECTION_MAP = {
    "LangSmith Deployment": "coll-deploy-001",
    "LangSmith Observability": "coll-obs-002",
    "LangSmith Evaluation": "coll-eval-003",
    "Troubleshooting": "coll-trouble-004",
    "Self Hosted": "coll-selfhost-005",
    "LangSmith Studio": "coll-studio-006",
    "General": "coll-general-007",
}

# ---------------------------------------------------------------------------
# Test cases
# Each tuple: (article_id, title, collection_id, expected_collection_name)
# The titles are designed to trigger the old keyword heuristics incorrectly.
# ---------------------------------------------------------------------------

CASES = [
    # Title has "langsmith" keyword but real collection is "LangSmith Deployment"
    (
        "art-001",
        "Setting up LangSmith in production",
        "coll-deploy-001",
        "LangSmith Deployment",
    ),
    # Title has "langsmith" keyword but real collection is "LangSmith Evaluation"
    (
        "art-002",
        "LangSmith evaluation datasets guide",
        "coll-eval-003",
        "LangSmith Evaluation",
    ),
    # Title does NOT contain "langsmith" or "langgraph" but belongs to "Troubleshooting"
    # (old heuristic would fall back to "Customer Support Knowledge Base")
    (
        "art-003",
        "Why is my agent slow?",
        "coll-trouble-004",
        "Troubleshooting",
    ),
    # Title has "self" and "host" — old heuristic maps to "Self Hosted" but
    # let's make sure the real collection_id is used (same result here, different path)
    (
        "art-004",
        "Self hosted deployment guide",
        "coll-selfhost-005",
        "Self Hosted",
    ),
    # Title has "langsmith" but real collection is "LangSmith Studio"
    (
        "art-005",
        "Debugging agents in LangSmith Studio",
        "coll-studio-006",
        "LangSmith Studio",
    ),
    # Title has "langsmith" but real collection is "LangSmith Observability"
    (
        "art-006",
        "LangSmith tracing overview",
        "coll-obs-002",
        "LangSmith Observability",
    ),
]


@pytest.mark.parametrize("article_id,title,collection_id,expected_collection", CASES)
def test_get_article_content_uses_collection_id(
    article_id, title, collection_id, expected_collection
):
    """get_article_content should use collection_id → name lookup, not title keywords."""
    mock_articles = [_make_article(article_id, title, collection_id)]

    import tools.pylon_tools as pylon_tools

    with (
        patch.object(pylon_tools, "_fetch_all_articles", return_value=mock_articles),
        patch.object(
            pylon_tools, "_fetch_collections", return_value=MOCK_COLLECTION_MAP
        ),
    ):
        # Call the underlying function (unwrap the @tool decorator if needed)
        func = getattr(
            pylon_tools.get_article_content, "func", pylon_tools.get_article_content
        )
        result = func(article_id)

    assert f"Collection: {expected_collection}" in result, (
        f"Expected 'Collection: {expected_collection}' in result, got:\n{result}"
    )


def test_get_article_content_fallback_on_unknown_collection_id():
    """When collection_id is not in the map, fall back to 'Customer Support Knowledge Base'."""
    mock_articles = [
        _make_article("art-999", "Some random article", "coll-unknown-999")
    ]

    import tools.pylon_tools as pylon_tools

    with (
        patch.object(pylon_tools, "_fetch_all_articles", return_value=mock_articles),
        patch.object(
            pylon_tools, "_fetch_collections", return_value=MOCK_COLLECTION_MAP
        ),
    ):
        func = getattr(
            pylon_tools.get_article_content, "func", pylon_tools.get_article_content
        )
        result = func("art-999")

    assert "Collection: Customer Support Knowledge Base" in result, (
        f"Expected fallback collection in result, got:\n{result}"
    )


def test_get_article_content_fallback_on_fetch_collections_exception():
    """When _fetch_collections() raises, fall back to 'Customer Support Knowledge Base'."""
    mock_articles = [_make_article("art-888", "LangSmith article", "coll-deploy-001")]

    import tools.pylon_tools as pylon_tools

    with (
        patch.object(pylon_tools, "_fetch_all_articles", return_value=mock_articles),
        patch.object(
            pylon_tools, "_fetch_collections", side_effect=Exception("API down")
        ),
    ):
        func = getattr(
            pylon_tools.get_article_content, "func", pylon_tools.get_article_content
        )
        result = func("art-888")

    assert "Collection: Customer Support Knowledge Base" in result, (
        f"Expected fallback collection in result, got:\n{result}"
    )
