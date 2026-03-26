"""Tests for search_support_articles() collection filtering in src/tools/pylon_tools.py.

These tests verify that "all" appearing in a comma-separated collections list is handled
gracefully (returns all articles instead of an error), and that the default "all" still works.

All HTTP calls and caches are mocked — no network access or API credentials required.
"""

import json
import unittest
from unittest.mock import patch

import src.tools.pylon_tools as pylon_module


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

MOCK_COLLECTIONS = {
    "OSS (LangChain and LangGraph)": "col-oss",
    "LangSmith Evaluation": "col-eval",
    "LangSmith Observability": "col-obs",
    "General": "col-gen",
}

# A minimal published, public article for each collection.
def _make_article(article_id, title, collection_id):
    return {
        "id": article_id,
        "title": title,
        "is_published": True,
        "visibility_config": {"visibility": "public"},
        "identifier": article_id,
        "slug": title.lower().replace(" ", "-"),
        "collection_id": collection_id,
    }


MOCK_ARTICLES = [
    _make_article("a1", "OSS Article One", "col-oss"),
    _make_article("a2", "OSS Article Two", "col-oss"),
    _make_article("a3", "Eval Article One", "col-eval"),
    _make_article("a4", "Obs Article One", "col-obs"),
]


# ---------------------------------------------------------------------------
# Helper — call the underlying function (bypassing @tool wrapper)
# ---------------------------------------------------------------------------

def _call_search(collections_arg):
    """Invoke search_support_articles with caches pre-populated."""
    pylon_module._articles_cache = MOCK_ARTICLES
    pylon_module._collections_cache = MOCK_COLLECTIONS
    # The @tool decorator wraps the function; call the underlying function directly.
    return pylon_module.search_support_articles.func(collections_arg)


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestSearchSupportArticlesCollectionFiltering(unittest.TestCase):
    """Unit tests for the "all" token handling in search_support_articles."""

    def setUp(self):
        """Reset caches before each test."""
        pylon_module._articles_cache = None
        pylon_module._collections_cache = None

    # ------------------------------------------------------------------
    # 1. collections="all" (default) — should return all articles
    # ------------------------------------------------------------------

    @patch("src.tools.pylon_tools._get_api_key", return_value="fake-key")
    @patch("src.tools.pylon_tools._get_kb_id", return_value="kb-123")
    def test_collections_all_returns_all_articles(self, mock_kb, mock_key):
        """collections='all' (default) must return every article without error."""
        result_str = _call_search("all")
        result = json.loads(result_str)

        self.assertNotIn("error", result, msg=f"Unexpected error: {result.get('error')}")
        self.assertEqual(result["total"], len(MOCK_ARTICLES))

    # ------------------------------------------------------------------
    # 2. collections="OSS (LangChain and LangGraph),all" — treat as all
    # ------------------------------------------------------------------

    @patch("src.tools.pylon_tools._get_api_key", return_value="fake-key")
    @patch("src.tools.pylon_tools._get_kb_id", return_value="kb-123")
    def test_oss_and_all_returns_all_articles(self, mock_kb, mock_key):
        """'OSS (LangChain and LangGraph),all' must return all articles, not an error."""
        result_str = _call_search("OSS (LangChain and LangGraph),all")
        result = json.loads(result_str)

        self.assertNotIn("error", result, msg=f"Unexpected error: {result.get('error')}")
        self.assertEqual(result["total"], len(MOCK_ARTICLES))

    # ------------------------------------------------------------------
    # 3. collections="LangSmith Evaluation,all" — treat as all
    # ------------------------------------------------------------------

    @patch("src.tools.pylon_tools._get_api_key", return_value="fake-key")
    @patch("src.tools.pylon_tools._get_kb_id", return_value="kb-123")
    def test_eval_and_all_returns_all_articles(self, mock_kb, mock_key):
        """'LangSmith Evaluation,all' must return all articles, not an error."""
        result_str = _call_search("LangSmith Evaluation,all")
        result = json.loads(result_str)

        self.assertNotIn("error", result, msg=f"Unexpected error: {result.get('error')}")
        self.assertEqual(result["total"], len(MOCK_ARTICLES))

    # ------------------------------------------------------------------
    # 4. Specific collection (no "all") — should still filter correctly
    # ------------------------------------------------------------------

    @patch("src.tools.pylon_tools._get_api_key", return_value="fake-key")
    @patch("src.tools.pylon_tools._get_kb_id", return_value="kb-123")
    def test_specific_collection_filters_correctly(self, mock_kb, mock_key):
        """A valid specific collection name must filter articles to that collection only."""
        result_str = _call_search("LangSmith Evaluation")
        result = json.loads(result_str)

        self.assertNotIn("error", result, msg=f"Unexpected error: {result.get('error')}")
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["articles"][0]["title"], "Eval Article One")

    # ------------------------------------------------------------------
    # 5. Unknown collection (no "all") — should return error
    # ------------------------------------------------------------------

    @patch("src.tools.pylon_tools._get_api_key", return_value="fake-key")
    @patch("src.tools.pylon_tools._get_kb_id", return_value="kb-123")
    def test_unknown_collection_returns_error(self, mock_kb, mock_key):
        """An unrecognised collection name (without 'all') must return an error."""
        result_str = _call_search("NonExistentCollection")
        result = json.loads(result_str)

        self.assertIn("error", result)
        self.assertIn("NonExistentCollection", result["error"])

    # ------------------------------------------------------------------
    # 6. "all" token alone in comma list, e.g. just "all,all"
    # ------------------------------------------------------------------

    @patch("src.tools.pylon_tools._get_api_key", return_value="fake-key")
    @patch("src.tools.pylon_tools._get_kb_id", return_value="kb-123")
    def test_all_repeated_returns_all_articles(self, mock_kb, mock_key):
        """'all,all' should be treated the same as 'all' — return all articles."""
        result_str = _call_search("all,all")
        result = json.loads(result_str)

        self.assertNotIn("error", result, msg=f"Unexpected error: {result.get('error')}")
        self.assertEqual(result["total"], len(MOCK_ARTICLES))

    # ------------------------------------------------------------------
    # 7. Case-insensitive "ALL" in list
    # ------------------------------------------------------------------

    @patch("src.tools.pylon_tools._get_api_key", return_value="fake-key")
    @patch("src.tools.pylon_tools._get_kb_id", return_value="kb-123")
    def test_all_uppercase_in_list_returns_all_articles(self, mock_kb, mock_key):
        """'OSS (LangChain and LangGraph),ALL' should work the same as including 'all'."""
        result_str = _call_search("OSS (LangChain and LangGraph),ALL")
        result = json.loads(result_str)

        self.assertNotIn("error", result, msg=f"Unexpected error: {result.get('error')}")
        self.assertEqual(result["total"], len(MOCK_ARTICLES))


if __name__ == "__main__":
    unittest.main()
