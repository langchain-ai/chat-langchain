"""Tests for support article identifier resolution."""

import unittest
from unittest.mock import patch

from src.tools import pylon_tools

ARTICLE = {
    "id": "uuid-123",
    "identifier": "6253531756",
    "slug": "state-backend-checkpointer",
    "title": "State backend and checkpointer",
    "collection_id": "collection-1",
    "current_published_content_html": "<p>Article content</p>",
}


class TestGetSupportArticleContent(unittest.TestCase):
    """Unit tests for support article content lookup."""

    def invoke(self, article_id):
        with (
            patch.object(pylon_tools, "_fetch_all_articles", return_value=[ARTICLE]),
            patch.object(
                pylon_tools,
                "_fetch_collections",
                return_value={"General": "collection-1"},
            ),
        ):
            return pylon_tools.get_support_article_content.invoke(
                {"article_id": article_id}
            )

    def test_uuid_hit(self):
        result = self.invoke("uuid-123")

        self.assertIn("ID: uuid-123", result)
        self.assertIn("Content:\n<p>Article content</p>", result)

    def test_bare_identifier_hit(self):
        result = self.invoke("6253531756")

        self.assertIn("ID: uuid-123", result)

    def test_identifier_slug_hit(self):
        result = self.invoke("6253531756-state-backend-checkpointer")

        self.assertIn("ID: uuid-123", result)

    def test_full_support_url_hit(self):
        result = self.invoke(
            " https://support.langchain.com/articles/6253531756-state-backend-checkpointer "
        )

        self.assertIn("ID: uuid-123", result)

    def test_unresolvable_id_returns_actionable_error(self):
        result = self.invoke("unknown-article")

        self.assertTrue(result.startswith("ERROR:"))
        self.assertIn(
            'required argument is the "id" (UUID) field from search_support_articles',
            result,
        )


if __name__ == "__main__":
    unittest.main()
