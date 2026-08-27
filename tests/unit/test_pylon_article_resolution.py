"""Tests for support article identifier resolution."""

import unittest
from unittest.mock import patch

from src.tools import pylon_tools

ARTICLES = [
    {
        "id": "uuid-123",
        "identifier": 1242226068,
        "slug": "how-do-i-configure-checkpointing-in-langgraph",
        "title": "Configure checkpointing",
        "current_published_content_html": "<p>Content</p>",
        "is_published": True,
        "visibility_config": {"visibility": "public"},
    },
]


class TestSupportArticleResolution(unittest.TestCase):
    """Test supported support article identifiers."""

    def test_uuid_hit(self):
        self.assertIs(pylon_tools._resolve_article(ARTICLES, "uuid-123"), ARTICLES[0])

    def test_numeric_identifier_hit(self):
        self.assertIs(
            pylon_tools._resolve_article(ARTICLES, "1242226068"), ARTICLES[0]
        )

    def test_identifier_slug_hit(self):
        self.assertIs(
            pylon_tools._resolve_article(
                ARTICLES, "1242226068-HOW-DO-I-CONFIGURE-CHECKPOINTING-IN-LANGGRAPH"
            ),
            ARTICLES[0],
        )

    def test_support_url_hit(self):
        self.assertIs(
            pylon_tools._resolve_article(
                ARTICLES,
                "https://support.langchain.com/articles/1242226068-how-do-i-configure-checkpointing-in-langgraph",
            ),
            ARTICLES[0],
        )

    @patch("src.tools.pylon_tools._fetch_collections", return_value={})
    @patch("src.tools.pylon_tools._fetch_all_articles", return_value=ARTICLES)
    def test_miss_is_self_correcting(self, mock_fetch_articles, mock_fetch_collections):
        result = pylon_tools.get_support_article_content.invoke(
            {"article_id": "not-a-real-article"}
        )

        self.assertIn("id` field returned by search_support_articles", result)
        self.assertIn("not its numeric identifier or URL slug", result)
        self.assertIn('"id": "uuid-123"', result)
        self.assertIn('"title": "Configure checkpointing"', result)


if __name__ == "__main__":
    unittest.main()
