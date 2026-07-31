"""Tests for article identifier resolution in src/tools/pylon_tools.py.

These tests do NOT require network access or LangSmith credentials.
_fetch_all_articles / _fetch_collections are mocked via unittest.mock.
"""

import unittest
from unittest.mock import patch

import src.tools.pylon_tools as pylon_module

ARTICLES = [
    {
        "id": "11111111-1111-1111-1111-111111111111",
        "identifier": "9051234",
        "slug": "how-to-trace-a-graph",
        "title": "How to trace a graph",
        "is_published": True,
        "visibility_config": {"visibility": "public"},
        "current_published_content_html": "<p>Tracing a graph</p>",
    },
    {
        "id": "22222222-2222-2222-2222-222222222222",
        "identifier": "9055678",
        "slug": "self-hosted-upgrades",
        "title": "Self hosted upgrades",
        "is_published": True,
        "visibility_config": {"visibility": "public"},
        "current_published_content_html": "<p>Upgrading self hosted</p>",
    },
    {
        "id": "33333333-3333-3333-3333-333333333333",
        "identifier": "9059999",
        "slug": "draft-article",
        "title": "Draft article",
        "is_published": False,
        "visibility_config": {"visibility": "private"},
        "current_published_content_html": "<p>Draft</p>",
    },
]


def _get_content(article_id):
    """Call the tool's underlying function directly."""
    return pylon_module.get_support_article_content.func(article_id)


class TestResolveArticle(unittest.TestCase):
    """Unit tests for _resolve_article()."""

    def test_resolves_by_uuid_id(self):
        article = pylon_module._resolve_article(
            ARTICLES, "11111111-1111-1111-1111-111111111111"
        )
        self.assertEqual(article["title"], "How to trace a graph")

    def test_resolves_by_numeric_identifier(self):
        article = pylon_module._resolve_article(ARTICLES, "9055678")
        self.assertEqual(article["title"], "Self hosted upgrades")

    def test_resolves_by_identifier_slug(self):
        article = pylon_module._resolve_article(ARTICLES, "9051234-how-to-trace-a-graph")
        self.assertEqual(article["title"], "How to trace a graph")

    def test_resolves_by_bare_slug(self):
        article = pylon_module._resolve_article(ARTICLES, "  Self-Hosted-Upgrades  ")
        self.assertEqual(article["title"], "Self hosted upgrades")

    def test_resolves_by_support_url(self):
        article = pylon_module._resolve_article(
            ARTICLES,
            "https://support.langchain.com/articles/9055678-self-hosted-upgrades/?utm=x#top",
        )
        self.assertEqual(article["title"], "Self hosted upgrades")

    def test_unknown_identifier_returns_none(self):
        self.assertIsNone(pylon_module._resolve_article(ARTICLES, "not-an-article"))
        self.assertIsNone(pylon_module._resolve_article(ARTICLES, ""))


@patch("src.tools.pylon_tools._fetch_collections", return_value={})
@patch("src.tools.pylon_tools._fetch_all_articles", return_value=ARTICLES)
class TestGetSupportArticleContent(unittest.TestCase):
    """Unit tests for get_support_article_content() resolution and miss path."""

    def test_uuid_id_returns_content(self, mock_articles, mock_collections):
        result = _get_content("11111111-1111-1111-1111-111111111111")
        self.assertIn("ID: 11111111-1111-1111-1111-111111111111", result)
        self.assertIn("Title: How to trace a graph", result)
        self.assertIn(
            "URL: https://support.langchain.com/articles/9051234-how-to-trace-a-graph",
            result,
        )
        self.assertIn("Tracing a graph", result)

    def test_numeric_identifier_returns_content(self, mock_articles, mock_collections):
        result = _get_content("9051234")
        self.assertIn("Title: How to trace a graph", result)

    def test_identifier_slug_returns_content(self, mock_articles, mock_collections):
        result = _get_content("9055678-self-hosted-upgrades")
        self.assertIn("Title: Self hosted upgrades", result)

    def test_support_url_returns_content(self, mock_articles, mock_collections):
        result = _get_content(
            "https://support.langchain.com/articles/9055678-self-hosted-upgrades"
        )
        self.assertIn("Title: Self hosted upgrades", result)

    def test_genuine_miss_raises_value_error(self, mock_articles, mock_collections):
        with self.assertRaises(ValueError) as ctx:
            _get_content("some-nonexistent-article")

        message = str(ctx.exception)
        self.assertIn('"id" field', message)
        self.assertIn("search_support_articles", message)
        self.assertIn("11111111-1111-1111-1111-111111111111", message)
        self.assertIn("22222222-2222-2222-2222-222222222222", message)
        # Unpublished articles are never suggested as candidates
        self.assertNotIn("33333333-3333-3333-3333-333333333333", message)


if __name__ == "__main__":
    unittest.main()
