"""Tests for get_article_content collection inference using collection_id.

These tests verify that get_article_content() correctly looks up the
collection name from the _fetch_collections() cache using article["collection_id"],
rather than guessing based on title keywords.
"""
import importlib
import sys
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MOCK_COLLECTIONS = {
    # name -> id  (mirrors what _fetch_collections() returns)
    "OSS (LangChain and LangGraph)": "coll-langgraph-oss",
    "LangSmith Observability": "coll-langsmith-obs",
    "LangSmith Deployment": "coll-langsmith-deploy",
    "Self Hosted": "coll-self-hosted",
    "General": "coll-general",
}


def _make_article(
    article_id: str,
    title: str,
    collection_id: str,
    identifier: str = "123",
    slug: str = "test-slug",
) -> dict:
    return {
        "id": article_id,
        "title": title,
        "collection_id": collection_id,
        "identifier": identifier,
        "slug": slug,
        "is_published": True,
        "visibility_config": {"visibility": "public"},
        "current_published_content_html": "<p>Some content</p>",
    }


# ---------------------------------------------------------------------------
# Fixtures / module reset
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_module_caches():
    """Reset pylon_tools module-level caches between tests."""
    # Ensure env vars are set so the module can be imported without error
    import os

    os.environ.setdefault("PYLON_KB_ID", "test-kb-id")
    os.environ.setdefault("PYLON_API_KEY", "test-api-key")

    # Re-import module to get fresh reference; reset caches
    import src.tools.pylon_tools as pylon_tools  # noqa: PLC0415

    pylon_tools._articles_cache = None
    pylon_tools._collections_cache = None
    yield
    pylon_tools._articles_cache = None
    pylon_tools._collections_cache = None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_collection_resolved_from_collection_id():
    """get_article_content returns the correct collection name from the cache."""
    import src.tools.pylon_tools as pylon_tools

    article = _make_article(
        article_id="art-001",
        title="How to deploy your application",  # No keywords -> old code would fall back
        collection_id="coll-langsmith-deploy",
    )

    with patch.object(pylon_tools, "_fetch_all_articles", return_value=[article]), patch.object(
        pylon_tools, "_fetch_collections", return_value=MOCK_COLLECTIONS
    ):
        result = pylon_tools.get_article_content.func("art-001")

    assert "Collection: LangSmith Deployment" in result


def test_langgraph_article_gets_correct_collection_not_keyword_guess():
    """A LangGraph article is labeled by its collection_id, not by title keyword."""
    import src.tools.pylon_tools as pylon_tools

    # Title contains "langgraph" -- old code would label this "LangGraph"
    # but the actual collection_id maps to "OSS (LangChain and LangGraph)"
    article = _make_article(
        article_id="art-002",
        title="Debugging LangGraph agents",
        collection_id="coll-langgraph-oss",
    )

    with patch.object(pylon_tools, "_fetch_all_articles", return_value=[article]), patch.object(
        pylon_tools, "_fetch_collections", return_value=MOCK_COLLECTIONS
    ):
        result = pylon_tools.get_article_content.func("art-002")

    assert "Collection: OSS (LangChain and LangGraph)" in result
    # Ensure the old incorrect keyword-derived names are NOT used
    assert "Collection: LangGraph\n" not in result


def test_fallback_when_collection_id_missing():
    """When an article has no collection_id the fallback label is used."""
    import src.tools.pylon_tools as pylon_tools

    article = _make_article(
        article_id="art-003",
        title="Generic question about LangSmith",
        collection_id=None,  # Missing collection_id
    )
    article["collection_id"] = None  # explicit None

    with patch.object(pylon_tools, "_fetch_all_articles", return_value=[article]), patch.object(
        pylon_tools, "_fetch_collections", return_value=MOCK_COLLECTIONS
    ):
        result = pylon_tools.get_article_content.func("art-003")

    assert "Collection: Customer Support Knowledge Base" in result


def test_fallback_when_collection_id_not_in_cache():
    """When collection_id exists but is not in the cache, the fallback label is used."""
    import src.tools.pylon_tools as pylon_tools

    article = _make_article(
        article_id="art-004",
        title="Some article",
        collection_id="coll-unknown-xyz",  # Not in MOCK_COLLECTIONS
    )

    with patch.object(pylon_tools, "_fetch_all_articles", return_value=[article]), patch.object(
        pylon_tools, "_fetch_collections", return_value=MOCK_COLLECTIONS
    ):
        result = pylon_tools.get_article_content.func("art-004")

    assert "Collection: Customer Support Knowledge Base" in result


def test_fallback_when_collections_fetch_fails():
    """When _fetch_collections() raises, the fallback label is still returned."""
    import src.tools.pylon_tools as pylon_tools

    article = _make_article(
        article_id="art-005",
        title="Some self host article",
        collection_id="coll-self-hosted",
    )

    with patch.object(pylon_tools, "_fetch_all_articles", return_value=[article]), patch.object(
        pylon_tools, "_fetch_collections", side_effect=RuntimeError("network error")
    ):
        result = pylon_tools.get_article_content.func("art-005")

    assert "Collection: Customer Support Knowledge Base" in result


def test_self_hosted_article_resolved_by_collection_id():
    """Self-hosted article resolves via collection_id, not via title keywords."""
    import src.tools.pylon_tools as pylon_tools

    article = _make_article(
        article_id="art-006",
        title="How to self host your instance",
        collection_id="coll-self-hosted",
    )

    with patch.object(pylon_tools, "_fetch_all_articles", return_value=[article]), patch.object(
        pylon_tools, "_fetch_collections", return_value=MOCK_COLLECTIONS
    ):
        result = pylon_tools.get_article_content.func("art-006")

    assert "Collection: Self Hosted" in result


def test_article_not_found():
    """Returns a helpful message when article_id does not match any article."""
    import src.tools.pylon_tools as pylon_tools

    article = _make_article("art-007", "Some article", "coll-general")

    with patch.object(pylon_tools, "_fetch_all_articles", return_value=[article]), patch.object(
        pylon_tools, "_fetch_collections", return_value=MOCK_COLLECTIONS
    ):
        result = pylon_tools.get_article_content.func("nonexistent-id")

    assert "not found" in result.lower()
