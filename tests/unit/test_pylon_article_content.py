"""Tests for support article content resolution."""

import importlib
import unittest
from unittest.mock import patch

from langchain_core.tools import ToolException

ARTICLE = {
    "id": "f1782122-2836-4a3e-b7c3-1a8c31bad015",
    "identifier": 6253531756,
    "slug": "state-backend-guide",
    "title": "State backend guide",
    "collection_id": "collection-1",
    "current_published_content_html": "<p>Article content</p>",
}


class TestGetSupportArticleContent(unittest.TestCase):
    """Unit tests for support article identifier resolution."""

    def setUp(self):
        """Load the tool module for each test."""
        self.module = importlib.import_module("src.tools.pylon_tools")

    def _fetch(self, article_id):
        with (
            patch.object(self.module, "_fetch_all_articles", return_value=[ARTICLE]),
            patch.object(self.module, "_fetch_collections", return_value={}),
        ):
            return self.module.get_support_article_content.invoke(
                {"article_id": article_id}
            )

    def test_resolves_by_uuid(self):
        result = self._fetch(ARTICLE["id"])
        self.assertIn("Article content", result)

    def test_resolves_by_numeric_identifier(self):
        result = self._fetch(str(ARTICLE["identifier"]))
        self.assertIn("Article content", result)

    def test_resolves_by_identifier_slug(self):
        result = self._fetch(f'{ARTICLE["identifier"]}-{ARTICLE["slug"]}'.upper())
        self.assertIn("Article content", result)

    def test_resolves_by_bare_slug(self):
        result = self._fetch(f'  {ARTICLE["slug"].upper()}  ')
        self.assertIn("Article content", result)

    def test_unknown_id_raises_tool_exception(self):
        with self.assertRaises(ToolException):
            self._fetch("unknown-article")


if __name__ == "__main__":
    unittest.main()
