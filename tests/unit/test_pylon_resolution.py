"""Tests for support article and collection resolution."""

import json
import unittest
from unittest.mock import patch

from src.tools import pylon_tools

ARTICLE = {
    "id": "uuid-123",
    "identifier": "6253531756",
    "slug": "stateful-agents",
    "title": "Stateful agents",
    "collection_id": "collection-1",
    "current_published_content_html": "<p>Content</p>",
}


class TestPylonResolution(unittest.TestCase):
    def setUp(self):
        pylon_tools._articles_cache = [ARTICLE]
        pylon_tools._article_index = pylon_tools._build_article_index([ARTICLE])

    @patch("src.tools.pylon_tools._fetch_collections", return_value={})
    def test_article_lookup_accepts_supported_identifiers(self, mock_collections):
        for article_id in (
            "uuid-123",
            "6253531756",
            "6253531756-stateful-agents",
            "https://support.langchain.com/articles/6253531756-stateful-agents/",
        ):
            with self.subTest(article_id=article_id):
                result = pylon_tools.get_support_article_content.invoke(
                    {"article_id": article_id}
                )
                self.assertIn("ID: uuid-123", result)
                self.assertIn("Content:\n<p>Content</p>", result)

    @patch("src.tools.pylon_tools._fetch_collections", return_value={})
    def test_article_not_found_message_is_actionable(self, mock_collections):
        result = pylon_tools.get_support_article_content.invoke(
            {"article_id": "stateful-agent"}
        )

        self.assertIn("was not recognized", result)
        self.assertIn("id=uuid-123 title=Stateful agents", result)
        self.assertIn("Retry with the id field from search_support_articles", result)

    @patch(
        "src.tools.pylon_tools._fetch_collections",
        return_value={"OSS (LangChain and LangGraph)": "collection-1"},
    )
    def test_search_matches_transposed_collection_tokens(self, mock_collections):
        article = {
            **ARTICLE,
            "is_published": True,
            "visibility_config": {"visibility": "public"},
        }
        pylon_tools._articles_cache = [article]

        result = json.loads(
            pylon_tools.search_support_articles.invoke(
                {"collections": "OSS (LangGraph and LangChain)"}
            )
        )

        self.assertEqual(result["total"], 1)
        self.assertEqual(result["warnings"], [])

    @patch(
        "src.tools.pylon_tools._fetch_collections",
        return_value={"Known": "collection-1"},
    )
    def test_search_returns_partial_results_with_warnings(self, mock_collections):
        article = {
            **ARTICLE,
            "is_published": True,
            "visibility_config": {"visibility": "public"},
        }
        pylon_tools._articles_cache = [article]

        result = json.loads(
            pylon_tools.search_support_articles.invoke(
                {"collections": "Known,Missing"}
            )
        )

        self.assertEqual(result["total"], 1)
        self.assertEqual(len(result["warnings"]), 1)
        self.assertIn("Missing", result["warnings"][0])


if __name__ == "__main__":
    unittest.main()
