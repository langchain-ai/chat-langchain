import json
from unittest.mock import patch

import pytest

from src.tools.pylon_tools import search_support_articles

COLLECTIONS = {
    "OSS (LangChain and LangGraph)": "oss-id",
    "LangSmith Observability": "observability-id",
}


ARTICLES = [
    {
        "id": "article-1",
        "title": "LangGraph support",
        "identifier": "article-1",
        "slug": "langgraph-support",
        "collection_id": "oss-id",
        "is_published": True,
        "visibility_config": {"visibility": "public"},
    },
    {
        "id": "article-2",
        "title": "Tracing support",
        "identifier": "article-2",
        "slug": "tracing-support",
        "collection_id": "observability-id",
        "is_published": True,
        "visibility_config": {"visibility": "public"},
    },
]


@patch("src.tools.pylon_tools._fetch_collections", return_value=COLLECTIONS)
@patch("src.tools.pylon_tools._fetch_all_articles", return_value=ARTICLES)
def test_reordered_collection_name_resolves(mock_articles, mock_collections):
    result = json.loads(
        search_support_articles.invoke("OSS (LangGraph and LangChain)")
    )

    assert result["total"] == 1
    assert result["articles"][0]["id"] == "article-1"
    assert result["unmatched_collections"] == []


@patch("src.tools.pylon_tools._fetch_collections", return_value=COLLECTIONS)
@patch("src.tools.pylon_tools._fetch_all_articles", return_value=ARTICLES)
def test_mixed_collection_names_keep_resolved_articles(
    mock_articles, mock_collections
):
    result = json.loads(
        search_support_articles.invoke("LangSmith Observability,Billing")
    )

    assert result["total"] == 1
    assert result["articles"][0]["id"] == "article-2"
    assert result["unmatched_collections"] == ["Billing"]
    assert result["available_collections"] == list(COLLECTIONS)


@patch("src.tools.pylon_tools._fetch_collections", return_value=COLLECTIONS)
@patch("src.tools.pylon_tools._fetch_all_articles", return_value=ARTICLES)
def test_all_bogus_collection_names_raise(
    mock_articles, mock_collections
):
    with pytest.raises(ValueError, match='collections="all"'):
        search_support_articles.invoke("Billing,Payments")
