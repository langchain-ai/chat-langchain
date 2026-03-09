"""Tests verifying that get_article_content strips HTML tags from output."""
import re
from unittest.mock import MagicMock, patch

import pytest


# ------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------

HTML_TAG_PATTERN = re.compile(r"<[^>]+>")


def _has_html_tags(text: str) -> bool:
    """Return True if the text contains any HTML tags."""
    return bool(HTML_TAG_PATTERN.search(text))


# ------------------------------------------------------------------------
# Fixtures / shared article data
# ------------------------------------------------------------------------

ARTICLE_ID = "art-001"

HTML_ARTICLE = {
    "id": ARTCICLE_ID,
    "title": "How to configure tracing",
    "is_published": True,
    "identifier": "12345",
    "slug": "how-to-configure-tracing",
    "visibility_config": {"visibility": "public"},
    "collection_id": "col-1",
    "current_published_content_html": (
        "<div class='article-body'>"
        "<p>This is a <strong>test</strong> article about "
        "<a href='https://docs.langchain.com'>LangChain</a>.</p>"
        "<ul><li>Step one: install the package</li>"
        "<li>Step two: configure your <span class='code'>LANGSMITH_API_KEY</span></li></ul>"
        "<p>For more information visit our <em>documentation</em>.</p>"
        "</div>"
    ),
}

PLAIN_TEXT_EXPECTED_SUBSTRINGS = [
    "test",
    "article about",
    "LangChain",
    "Step one",
    "Step two",
    "configure your",
    "LANGSMITH_API_KEY",
    "documentation",
]

HTML_TAGS_NOT_EXPECTED = [
    "<div",
    "<p>",
    "</p>",
    "<strong>",
    "</strong>",
    "<a href",
    "</a>",
    "<ul>",
    "<li>",
    "</li>",
    "</ul>",
    "<span",
    "</span>",
    "<em>",
    "</em>",
]


# ------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------


def test_get_article_content_strips_html():
    """get_article_content should return plain text without any HTML tags."""
    with patch(
        "src.tools.pylon_tools._fetch_all_articles", return_value=[HTML_ARTICLE]
    ):
        from src.tools.pylon_tools import get_article_content

        result = get_article_content.invoke({"article_id": ARTICLE_ID})

    assert not _has_html_tags(result), (
        f"Output still contains HTML tags. Got:\n{result}"
    )


def test_get_article_content_no_p_tags():
    """get_article_content output must not contain <p> or </p> tags."""
    with patch(
        "src.tools.pylon_tools._fetch_all_articles", return_value=[HTML_ARTICLE]
    ):
        from src.tools.pylon_tools import get_article_content

        result = get_article_content.invoke({"article_id": ARTICLE_ID})

    assert "<p>" not in result, "Output contains <p> tag"
    assert "</p>" not in result, "Output contains </p> tag"


def test_get_article_content_no_div_tags():
    """get_article_content output must not contain <div> tags."""
    with patch(
        "src.tools.pylon_tools._fetch_all_articles", return_value=[HTML_ARTICLE]
    ):
        from src.tools.pylon_tools import get_article_content

        result = get_article_content.invoke({"article_id": ARTICLE_ID})

    assert "<div" not in result, "Output contains <div> tag"


def test_get_article_content_no_anchor_tags():
    """get_article_content output must not contain <a href=...> tags."""
    with patch(
        "src.tools.pylon_tools._fetch_all_articles", return_value=[HTML_ARTICLE]
    ):
        from src.tools.pylon_tools import get_article_content

        result = get_article_content.invoke({"article_id": ARTICLE_ID})

    assert "<a href" not in result, "Output contains <a href> anchor tag"
    assert "</a>" not in result, "Output contains </a> closing anchor tag"


def test_get_article_content_no_span_tags():
    """get_article_content output must not contain <span> tags."""
    with patch(
        "src.tools.pylon_tools._fetch_all_articles", return_value=[HTML_ARTICLE]
    ):
        from src.tools.pylon_tools import get_article_content

        result = get_article_content.invoke({"article_id": ARTCICLE_ID})

    assert "<span" not in result, "Output contains <span> tag"


def test_get_article_content_preserves_text_content():
    """get_article_content should keep the actual text content after stripping."""
    with patch(
        "src.tools.pylon_tools._fetch_all_articles", return_value=[HTML_ARTICLE]
    ):
        from src.tools.pylon_tools import get_article_content

        result = get_article_content.invoke({"article_id": ARTCICLE_ID})

    for substring in PLAIN_TEXT_EXPECTED_SUBSTRINGS:
        assert substring in result, (
            f"Expected text '{substring}' missing from output. Got:\n{result}"
        )


def test_get_article_content_all_html_tags_absent():
    """Comprehensive check: none of the known HTML tags should appear in output."""
    with patch(
        "src.tools.pylon_tools._fetch_all_articles", return_value=[HTML_ARTICLE]
    ):
        from src.tools.pylon_tools import get_article_content

        result = get_article_content.invoke({"article_id": ARTICLE_ID})

    for tag in HTML_TAGS_NOT_EXPECTED:
        assert tag not in result, f"Output still contains HTML tag '{tag}': {result}"


def test_get_article_content_truncation_applies_to_plain_text():
    """Truncation limit (5000 chars) should apply to plain text, not raw HTML."""
    # Build a long HTML article whose raw HTML is > 5000 chars but whose
    # plain text is also > 5000 chars -- after stripping, the output should be
    # plain text truncated at 5000 chars (no HTML tags anywhere).
    long_paragraph = "<p>" + ("word " * 1000) + "</p>"  # ~5010 chars raw
    long_article = dict(HTML_ARTICLE)
    long_article["current_published_content_html"] = long_paragraph

    with patch(
        "src.tools.pylon_tools._fetch_all_articles", return_value=[long_article]
    ):
        from src.tools.pylon_tools import get_article_content

        result = get_article_content.invoke({"article_id": ARTCICLE_ID})

    # Output must not have HTML tags regardless of length
    assert not _has_html_tags(result), (
        "Output from long article still contains HTML tags"
    )
    # The content section itself should contain plain words, not HTML
    assert "word" in result


def test_get_article_content_strips_nested_html():
    """get_article_content strips deeply nested HTML structures."""
    nested_html_article = dict(HTML_ARTICLE)
    nested_html_article["current_published_content_html"] = (
        "<div><section><article><p>"
        "<strong><em>Important:</em></strong> nested content here"
        "</p></article></section></div>"
    )

    with patch(
        "src.tools.pylon_tools._fetch_all_articles",
        return_value=[nested_html_article],
    ):
        from src.tools.pylon_tools import get_article_content

        result = get_article_content.invoke({"article_id": ARTCICLE_ID})

    assert not _has_html_tags(result), "Deeply nested HTML was not stripped"
    assert "Important" in result, "Text content was lost after stripping"
    assert "nested content here" in result, "Text content was lost after stripping"
