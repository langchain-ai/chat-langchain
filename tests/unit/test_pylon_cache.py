"""Tests for the TTL cache around Pylon KB articles and collections."""

import json
import unittest
from unittest.mock import MagicMock, patch

import requests

import src.tools.pylon_tools as pylon


def _response(data):
    resp = MagicMock()
    resp.json.return_value = {"data": data}
    resp.raise_for_status.return_value = None
    return resp


def _article(article_id):
    return {
        "id": article_id,
        "title": f"Article {article_id}",
        "is_published": True,
        "visibility_config": {"visibility": "public"},
        "identifier": article_id,
        "slug": "slug",
        "collection_id": "c1",
        "current_published_content_html": f"<p>{article_id}</p>",
    }


COLLECTIONS = [
    {"id": "c1", "title": "General", "visibility_config": {"visibility": "public"}}
]


@patch("src.tools.pylon_tools.PYLON_CACHE_TTL_SECONDS", 900)
@patch("src.tools.pylon_tools._get_api_key", return_value="fake-key")
@patch("src.tools.pylon_tools._get_kb_id", return_value="kb-123")
@patch("src.tools.pylon_tools.time.monotonic")
@patch("src.tools.pylon_tools.requests.get")
class TestPylonCacheTTL(unittest.TestCase):
    def setUp(self):
        pylon._articles_cache.clear()
        pylon._collections_cache.clear()

    def test_cache_hit_within_ttl(self, mock_get, mock_time, *_):
        mock_get.return_value = _response([_article("a1")])
        mock_time.return_value = 1000.0
        first = pylon._fetch_all_articles()
        mock_time.return_value = 1000.0 + 899
        second = pylon._fetch_all_articles()

        self.assertEqual(first, second)
        mock_get.assert_called_once()

    def test_refetch_after_ttl(self, mock_get, mock_time, *_):
        mock_get.side_effect = [
            _response([_article("a1")]),
            _response([_article("a1"), _article("a2")]),
        ]
        mock_time.return_value = 1000.0
        self.assertEqual(len(pylon._fetch_all_articles()), 1)
        mock_time.return_value = 1000.0 + 901
        self.assertEqual(len(pylon._fetch_all_articles()), 2)
        self.assertEqual(mock_get.call_count, 2)

    def test_stale_cache_served_when_refetch_fails(self, mock_get, mock_time, *_):
        mock_get.side_effect = [
            _response([_article("a1")]),
            requests.exceptions.ConnectionError("pylon down"),
        ]
        mock_time.return_value = 1000.0
        first = pylon._fetch_all_articles()
        mock_time.return_value = 1000.0 + 901
        with self.assertLogs("src.tools.pylon_tools", level="WARNING"):
            second = pylon._fetch_all_articles()

        self.assertEqual(first, second)
        self.assertEqual(mock_get.call_count, 2)

    def test_failed_refresh_not_retried_until_next_ttl(self, mock_get, mock_time, *_):
        mock_get.side_effect = [
            _response([_article("a1")]),
            requests.exceptions.ConnectionError("pylon down"),
            _response([_article("a1"), _article("a2")]),
        ]
        mock_time.return_value = 1000.0
        pylon._fetch_all_articles()
        mock_time.return_value = 1000.0 + 901
        with self.assertLogs("src.tools.pylon_tools", level="WARNING"):
            pylon._fetch_all_articles()
        mock_time.return_value = 1000.0 + 901 + 899
        self.assertEqual(len(pylon._fetch_all_articles()), 1)
        self.assertEqual(mock_get.call_count, 2)
        mock_time.return_value = 1000.0 + 901 + 901
        self.assertEqual(len(pylon._fetch_all_articles()), 2)
        self.assertEqual(mock_get.call_count, 3)

    def test_stale_served_without_waiting_while_refresh_runs(
        self, mock_get, mock_time, *_
    ):
        mock_get.return_value = _response([_article("a1")])
        mock_time.return_value = 1000.0
        first = pylon._fetch_all_articles()
        mock_time.return_value = 1000.0 + 901
        with pylon._articles_cache.lock:
            second = pylon._fetch_all_articles()

        self.assertEqual(first, second)
        mock_get.assert_called_once()

    def test_failure_without_cache_raises(self, mock_get, mock_time, *_):
        mock_get.side_effect = requests.exceptions.ConnectionError("pylon down")
        mock_time.return_value = 1000.0
        with self.assertRaises(requests.exceptions.ConnectionError):
            pylon._fetch_all_articles()

    def test_collections_refetch_after_ttl(self, mock_get, mock_time, *_):
        renamed = [dict(COLLECTIONS[0], title="Renamed")]
        mock_get.side_effect = [_response(COLLECTIONS), _response(renamed)]
        mock_time.return_value = 1000.0
        self.assertEqual(pylon._fetch_collections(), {"General": "c1"})
        self.assertEqual(pylon._fetch_collections(), {"General": "c1"})
        mock_time.return_value = 1000.0 + 901
        self.assertEqual(pylon._fetch_collections(), {"Renamed": "c1"})
        self.assertEqual(mock_get.call_count, 2)

    def test_new_article_searchable_and_fetchable_after_ttl(
        self, mock_get, mock_time, *_
    ):
        def fake_get(url, **kwargs):
            if url.endswith("/collections"):
                return _response(COLLECTIONS)
            return _response(articles)

        mock_get.side_effect = fake_get
        articles = [_article("a1")]
        mock_time.return_value = 1000.0
        result = json.loads(pylon.search_support_articles.invoke({}))
        self.assertEqual([a["id"] for a in result["articles"]], ["a1"])

        articles = [_article("a1"), _article("a2")]
        mock_time.return_value = 1000.0 + 901
        result = json.loads(pylon.search_support_articles.invoke({}))
        self.assertEqual([a["id"] for a in result["articles"]], ["a1", "a2"])
        content = pylon.get_support_article_content.invoke({"article_id": "a2"})
        self.assertIn("<p>a2</p>", content)


if __name__ == "__main__":
    unittest.main()
