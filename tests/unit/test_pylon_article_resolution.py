"""Tests for Pylon support article identifier resolution."""

import unittest
from unittest.mock import patch

ARTICLE = {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "identifier": 1234567890,
    "slug": "configure-langsmith",
    "title": "Configure LangSmith",
    "collection_id": "collection-1",
    "current_published_content_html": "Article content",
}


class TestPylonArticleResolution(unittest.TestCase):
    def test_all_supported_identifier_forms_resolve_to_same_article(self):
        from src.tools import pylon_tools

        identifiers = [
            ARTICLE["id"],
            str(ARTICLE["identifier"]),
            "1234567890-configure-langsmith",
            "https://support.langchain.com/articles/1234567890-configure-langsmith",
        ]

        with patch.object(pylon_tools, "_fetch_all_articles", return_value=[ARTICLE]), patch.object(
            pylon_tools, "_fetch_collections", return_value={}
        ):
            results = [
                pylon_tools.get_support_article_content.invoke({"article_id": value})
                for value in identifiers
            ]

        self.assertEqual(results, [results[0]] * len(identifiers))
        self.assertIn("ID: 550e8400-e29b-41d4-a716-446655440000", results[0])

    def test_unknown_identifier_explains_accepted_forms(self):
        from src.tools import pylon_tools

        with patch.object(pylon_tools, "_fetch_all_articles", return_value=[ARTICLE]):
            result = pylon_tools.get_support_article_content.invoke(
                {"article_id": "unknown-article"}
            )

        self.assertIn("article UUID, numeric identifier, or identifier-slug", result)
        self.assertIn("550e8400-e29b-41d4-a716-446655440000: Configure LangSmith", result)


if __name__ == "__main__":
    unittest.main()
