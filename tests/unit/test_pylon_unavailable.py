"""Pylon KB tool errors must tell the model the support KB is unavailable."""

import json
import unittest
from unittest.mock import MagicMock, patch

import requests

import src.tools.pylon_tools as pylon


def _http_error(status):
    resp = MagicMock()
    resp.status_code = status
    return requests.exceptions.HTTPError(
        f"{status} Client Error: Unauthorized for url: https://api.usepylon.com/x",
        response=resp,
    )


@patch("src.tools.pylon_tools._get_api_key", return_value="fake-key")
@patch("src.tools.pylon_tools._get_kb_id", return_value="kb-123")
class TestPylonUnavailable(unittest.TestCase):
    def setUp(self):
        pylon._articles_cache.clear()
        pylon._collections_cache.clear()

    def _search(self, error):
        with patch("src.tools.pylon_tools._fetch_all_articles", side_effect=error):
            return json.loads(pylon.search_support_articles.invoke({"collections": "all"}))

    def test_search_reports_unavailable_on_401(self, *_):
        out = self._search(_http_error(401))
        self.assertTrue(out["support_kb_unavailable"])
        self.assertEqual(out["status"], 401)
        self.assertIn("Do not conclude that no support article exists", out["note"])
        self.assertIn("do not mention the support knowledge base", out["note"])

    def test_search_reports_unavailable_on_connection_error(self, *_):
        out = self._search(requests.exceptions.ConnectionError("boom"))
        self.assertTrue(out["support_kb_unavailable"])
        self.assertIsNone(out["status"])

    def test_get_article_reports_unavailable_on_401(self, *_):
        with patch("src.tools.pylon_tools._fetch_all_articles", side_effect=_http_error(401)):
            out = json.loads(pylon.get_support_article_content.invoke({"article_id": "a1"}))
        self.assertTrue(out["support_kb_unavailable"])
        self.assertEqual(out["status"], 401)


if __name__ == "__main__":
    unittest.main()
