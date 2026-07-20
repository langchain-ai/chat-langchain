"""Tests that a get_support_article_content miss returns a distinct error signal.

These tests do NOT require network access or LangSmith credentials.
"""

import unittest
from unittest.mock import patch

import src.tools.pylon_tools as pylon_module


class TestArticleMissSignal(unittest.TestCase):
    """A knowledge-base miss must be a recognizable error, not a success string."""

    @patch("src.tools.pylon_tools._fetch_collections", return_value={})
    @patch(
        "src.tools.pylon_tools._fetch_all_articles",
        return_value=[{"id": "a1", "title": "Article 1"}],
    )
    def test_miss_returns_error_flagged_signal(self, mock_articles, mock_collections):
        result = pylon_module.get_support_article_content.invoke(
            {"article_id": "does-not-exist"}
        )
        self.assertIn("ERROR_ARTICLE_NOT_FOUND", result)
        self.assertIn("does-not-exist", result)
        self.assertNotIn("not found in knowledge base.", result)


if __name__ == "__main__":
    unittest.main()
