"""Tests for support article key normalization and lookup."""

import importlib
import json
import unittest
from unittest.mock import patch

ARTICLE = {
    "id": "uuid-123",
    "identifier": "1242226068",
    "slug": "how-do-i-configure-checkpointing-in-langgraph",
    "title": "How do I configure checkpointing in LangGraph?",
    "collection_id": "collection-1",
    "is_published": True,
    "visibility_config": {"visibility": "public"},
    "current_published_content_html": "<p>Article content</p>",
}


class TestPylonArticleLookup(unittest.TestCase):
    """Unit tests for support article lookup keys."""

    def setUp(self):
        """Reload the module and reset its caches before each test."""
        import src.tools.pylon_tools as pylon_module

        pylon_module = importlib.reload(pylon_module)
        pylon_module._articles_cache = [ARTICLE]
        pylon_module._collections_cache = {"Support": "collection-1"}
        self.module = pylon_module

    def test_all_supported_key_forms_resolve_article(self):
        """Each supported key form resolves to the same article."""
        keys = [
            ARTICLE["id"],
            ARTICLE["identifier"],
            ARTICLE["slug"],
            f'{ARTICLE["identifier"]}-{ARTICLE["slug"]}',
            f' https://support.langchain.com/articles/{ARTICLE["identifier"]}-{ARTICLE["slug"]} ',
        ]

        for key in keys:
            with self.subTest(key=key):
                result = self.module.get_support_article_content.invoke(
                    {"article_id": key}
                )
                self.assertIn("ID: uuid-123", result)
                self.assertIn("Article content", result)

    def test_not_found_message_names_accepted_fields(self):
        """Not-found errors explain which article keys are accepted."""
        result = self.module.get_support_article_content.invoke(
            {"article_id": "1242226068-wrong-slug"}
        )

        self.assertIn("id, identifier, slug, identifier-slug", result)
        self.assertIn("uuid-123 — How do I configure checkpointing in LangGraph?", result)

    @patch("src.tools.pylon_tools._fetch_collections", return_value={})
    def test_search_emits_article_id_and_guidance(self, mock_collections):
        """Search results label the UUID separately from the article URL."""
        result = self.module.search_support_articles.invoke({"collections": "all"})
        payload = json.loads(result)

        self.assertEqual(payload["articles"][0]["article_id"], ARTICLE["id"])
        self.assertNotIn("id", payload["articles"][0])
        self.assertIn("article_id", payload["note"])
        self.assertIn("slug inside url", payload["note"])


if __name__ == "__main__":
    unittest.main()
