"""Unit tests for get_support_article_content provenance validation."""

from unittest.mock import patch

import pytest

from src.tools.pylon_tools import get_support_article_content


@pytest.mark.parametrize(
    "article_id", ["0", "N/A", "00000000-0000-0000-0000-000000000000"]
)
def test_rejects_invalid_article_ids(article_id):
    """Invalid article IDs return a corrective provenance error without a KB call."""
    with patch("src.tools.pylon_tools._fetch_all_articles") as fetch_articles:
        result = get_support_article_content.invoke({"article_id": article_id})

    assert "search_support_articles" in result
    assert "articles[].id" in result
    fetch_articles.assert_not_called()


def test_accepts_uuid_shape():
    """Valid UUIDs proceed to the knowledge-base lookup."""
    article_id = "123e4567-e89b-12d3-a456-426614174000"
    with patch("src.tools.pylon_tools._fetch_all_articles", return_value=[]):
        result = get_support_article_content.invoke({"article_id": article_id})

    assert (
        result
        == "Error: No articles available from API. Check PYLON_API_KEY configuration."
    )
