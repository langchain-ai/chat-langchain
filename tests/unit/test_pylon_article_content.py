"""Tests for support article ID resolution and not-found guidance."""

from unittest.mock import patch

from src.tools.pylon_tools import get_support_article_content

ARTICLES = [
    {
        "id": "uuid-otel",
        "identifier": 7335403634,
        "slug": "how-do-i-use-opentelemetry-otel-with-langsmith",
        "title": "How do I use OpenTelemetry (OTEL) with LangSmith?",
        "collection_id": "collection-1",
        "current_published_content_html": "<p>OTEL content</p>",
    },
    {
        "id": "uuid-deploy",
        "identifier": 6253531756,
        "slug": "deployment-guide",
        "title": "Deployment guide",
        "collection_id": "collection-1",
        "current_published_content_html": "<p>Deployment content</p>",
    },
]


def _get_content(article_id: str) -> str:
    with (
        patch("src.tools.pylon_tools._fetch_all_articles", return_value=ARTICLES),
        patch(
            "src.tools.pylon_tools._fetch_collections",
            return_value={"Support": "collection-1"},
        ),
    ):
        return get_support_article_content.func(article_id)


def test_uuid_lookup_succeeds():
    result = _get_content("uuid-otel")

    assert "ID: uuid-otel" in result
    assert "OTEL content" in result


def test_numeric_identifier_lookup_succeeds():
    result = _get_content("7335403634")

    assert "ID: uuid-otel" in result


def test_identifier_slug_lookup_succeeds():
    result = _get_content("7335403634-how-do-i-use-opentelemetry-otel-with-langsmith")

    assert "ID: uuid-otel" in result


def test_full_support_url_lookup_succeeds():
    result = _get_content(
        "https://support.langchain.com/articles/7335403634-"
        "how-do-i-use-opentelemetry-otel-with-langsmith"
    )

    assert "ID: uuid-otel" in result


def test_unresolvable_id_returns_guidance_message():
    result = _get_content("opentelemetry")

    assert "did not resolve" in result
    assert "id" in result.lower()
    assert "uuid-otel" in result
    assert "How do I use OpenTelemetry (OTEL) with LangSmith?" in result
