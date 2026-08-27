"""Tests for support article identifier resolution."""

import unittest
from unittest.mock import patch

ARTICLE = {
    "id": "11111111-2222-3333-4444-555555555555",
    "identifier": "6253531756",
    "slug": "understanding-checkpointers-databases-api-memory-and-ttl",
    "title": "Understanding checkpointers, databases, API memory, and TTL",
    "collection_id": "collection-1",
    "current_published_content_html": "Article content",
}


class TestGetSupportArticleContent(unittest.TestCase):
    """Test support article identifier resolution."""

    def setUp(self):
        import src.tools.pylon_tools as pylon_module

        self.module = pylon_module
        self.url = (
            "https://support.langchain.com/articles/"
            "6253531756-understanding-checkpointers-databases-api-memory-and-ttl"
        )

    def _invoke(self, article_id):
        with patch.object(self.module, "_fetch_all_articles", return_value=[ARTICLE]), patch.object(
            self.module, "_fetch_collections", return_value={"General": "collection-1"}
        ):
            return self.module.get_support_article_content.invoke(
                {"article_id": article_id}
            )

    def test_uuid_resolves(self):
        result = self._invoke(ARTICLE["id"])

        self.assertIn("ID: " + ARTICLE["id"], result)
        self.assertIn("Article content", result)

    def test_full_url_slug_resolves(self):
        result = self._invoke("  " + self.url + "  ")

        self.assertIn("ID: " + ARTICLE["id"], result)

    def test_numeric_prefix_resolves(self):
        result = self._invoke(ARTICLE["identifier"])

        self.assertIn("ID: " + ARTICLE["id"], result)

    def test_unknown_id_returns_recovery_candidates(self):
        result = self._invoke("unknown-article")

        self.assertIn("id", result.lower())
        self.assertIn(ARTICLE["id"], result)


if __name__ == "__main__":
    unittest.main()
