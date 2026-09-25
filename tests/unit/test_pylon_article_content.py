"""Tests for support article identifier validation and lookup recovery."""

import unittest
from unittest.mock import patch

VALID_UUID = "12345678-1234-1234-1234-123456789abc"


def _article(article_id=VALID_UUID, identifier="12345", slug="getting-started"):
    return {
        "id": article_id,
        "identifier": identifier,
        "slug": slug,
        "title": "Getting Started",
        "current_published_content_html": "<p>Content</p>",
        "collection_id": "collection-1",
    }


class TestGetSupportArticleContent(unittest.TestCase):
    """Unit tests for get_support_article_content()."""

    @patch("src.tools.pylon_tools._fetch_collections", return_value={})
    @patch("src.tools.pylon_tools._fetch_all_articles")
    def test_placeholder_rejected_without_scanning(
        self, mock_fetch_articles, mock_fetch_collections
    ):
        from src.tools.pylon_tools import get_support_article_content

        result = get_support_article_content.invoke({"article_id": "dummy_id"})

        self.assertEqual(
            result,
            "Invalid article_id: 'dummy_id' is a placeholder, not a real "
            "identifier. Do not invent article IDs. Call search_support_articles "
            'and pass the exact value of an article\'s "id" field.',
        )
        mock_fetch_articles.assert_not_called()
        mock_fetch_collections.assert_not_called()

    @patch("src.tools.pylon_tools._fetch_collections", return_value={})
    @patch("src.tools.pylon_tools._fetch_all_articles", return_value=[_article()])
    def test_valid_uuid_returns_article_content(
        self, mock_fetch_articles, mock_fetch_collections
    ):
        from src.tools.pylon_tools import get_support_article_content

        result = get_support_article_content.invoke({"article_id": VALID_UUID})

        self.assertIn(f"ID: {VALID_UUID}", result)
        self.assertIn("Content:\n<p>Content</p>", result)
        mock_fetch_articles.assert_called_once_with()

    @patch("src.tools.pylon_tools._fetch_collections", return_value={})
    @patch("src.tools.pylon_tools._fetch_all_articles", return_value=[_article()])
    def test_unique_numeric_slug_recovers_article(
        self, mock_fetch_articles, mock_fetch_collections
    ):
        from src.tools.pylon_tools import get_support_article_content

        result = get_support_article_content.invoke(
            {"article_id": "12345-getting-started"}
        )

        self.assertIn(f"ID: {VALID_UUID}", result)

    @patch("src.tools.pylon_tools._fetch_collections", return_value={})
    @patch(
        "src.tools.pylon_tools._fetch_all_articles",
        return_value=[_article(), _article(VALID_UUID.replace("a", "b"))],
    )
    def test_multiple_numeric_matches_return_url_fragment_error(
        self, mock_fetch_articles, mock_fetch_collections
    ):
        from src.tools.pylon_tools import get_support_article_content

        result = get_support_article_content.invoke({"article_id": "12345-other"})

        self.assertEqual(
            result,
            "Invalid article_id: '12345-other' looks like a URL fragment, not "
            'the article "id" field. Call search_support_articles and pass the '
            'exact UUID from the "id" field of the result you want.',
        )

    @patch("src.tools.pylon_tools._fetch_collections", return_value={})
    @patch(
        "src.tools.pylon_tools._fetch_all_articles",
        return_value=[_article("87654321-4321-4321-4321-abcdefabcdef")],
    )
    def test_absent_valid_uuid_returns_distinct_error(
        self, mock_fetch_articles, mock_fetch_collections
    ):
        from src.tools.pylon_tools import get_support_article_content

        result = get_support_article_content.invoke({"article_id": VALID_UUID})

        self.assertEqual(
            result,
            f"Article {VALID_UUID} is a valid ID but is not present in the knowledge "
            "base. Re-run search_support_articles to get current article IDs.",
        )

    @patch("src.tools.pylon_tools._fetch_collections", return_value={})
    @patch("src.tools.pylon_tools._fetch_all_articles", return_value=[_article()])
    def test_zero_numeric_matches_return_url_fragment_error(
        self, mock_fetch_articles, mock_fetch_collections
    ):
        from src.tools.pylon_tools import get_support_article_content

        result = get_support_article_content.invoke({"article_id": "99999-unknown"})

        self.assertEqual(
            result,
            "Invalid article_id: '99999-unknown' looks like a URL fragment, not "
            'the article "id" field. Call search_support_articles and pass the '
            'exact UUID from the "id" field of the result you want.',
        )


if __name__ == "__main__":
    unittest.main()
