"""Test support article conversion and truncation without network access."""

import unittest
from unittest.mock import patch

from src.tools.pylon_tools import (
    ARTICLE_CONTENT_MAX_CHARS,
    _format_article_content,
    get_support_article_content,
)


class TestArticleContent(unittest.TestCase):
    def test_entities_links_and_blocks(self):
        html = (
            '<h2>Setup &amp; usage</h2><p>Read <a href="https://example.com/?a=1&amp;b=2">'
            "the <strong>guide</strong></a>.</p><p>&lt;ready&gt; &#169;</p>"
        )
        self.assertEqual(
            _format_article_content(html),
            "Setup & usage\n\nRead [the guide](https://example.com/?a=1&b=2).\n\n<ready> ©",
        )

    def test_code_preserves_whitespace_and_entities(self):
        html = '<pre><code>if value &lt; 3:\n    print("a &amp; b")\n</code></pre>'
        self.assertEqual(
            _format_article_content(html),
            '```\nif value < 3:\n    print("a & b")\n```',
        )
        self.assertEqual(
            _format_article_content('<code>print("```")</code>'),
            '````\nprint("```")\n````',
        )

    def test_under_and_at_cap_have_no_marker(self):
        for length in (0, 100, ARTICLE_CONTENT_MAX_CHARS):
            with self.subTest(length=length):
                text = "x" * length
                self.assertEqual(_format_article_content(f"<p>{text}</p>"), text)

    def test_limit_applies_to_converted_text(self):
        html = '<span data-long="' + "x" * 13000 + '">Complete instructions</span>'
        self.assertEqual(_format_article_content(html), "Complete instructions")

    def test_over_limit_reports_exact_counts(self):
        cap = ARTICLE_CONTENT_MAX_CHARS
        text = "x" * (cap + 123)
        self.assertEqual(
            _format_article_content(f"<p>{text}</p>"),
            f"{'x' * cap}\n\n[Article truncated: showing {cap} of {len(text)} characters]",
        )

    def test_old_cut_inside_attribute_returns_readable_marked_content(self):
        cap = ARTICLE_CONTENT_MAX_CHARS
        html = '<p data-long="' + "x" * 6000 + '">' + "Step. " * 2500 + "</p>"
        self.assertEqual(html[4999], "x")
        converted = ("Step. " * 2500).strip()
        self.assertEqual(
            _format_article_content(html),
            f"{converted[:cap]}\n\n[Article truncated: showing {cap} of {len(converted)} characters]",
        )

    def test_cut_does_not_split_link(self):
        cap = ARTICLE_CONTENT_MAX_CHARS
        prefix = "x" * (cap - 10)
        link = "[guide](https://example.com/long-target)"
        html = f'{prefix}<a href="https://example.com/long-target">guide</a>'
        self.assertEqual(
            _format_article_content(html),
            f"{prefix}\n\n[Article truncated: showing {len(prefix)} of {len(prefix + link)} characters]",
        )

    def test_cut_closes_code_fence(self):
        cap = ARTICLE_CONTENT_MAX_CHARS
        code = "x" * (cap + 100)
        result = _format_article_content(f"<pre><code>{code}</code></pre>")
        self.assertEqual(
            result,
            f"```\n{'x' * (cap - 8)}\n```\n\n[Article truncated: showing {cap} of {len(code) + 8} characters]",
        )

    def test_hidden_html_is_not_article_text(self):
        self.assertEqual(
            _format_article_content(
                "<style>p {color: red}</style><p>Read<br>this</p><script>bad()</script>"
            ),
            "Read\nthis",
        )

    def test_code_boundary_cuts_remain_balanced_with_exact_counts(self):
        html = "prefix<pre><code>" + "x" * 20 + "</code></pre>tail"
        converted = "prefix\n\n```\n" + "x" * 20 + "\n```\n\ntail"
        for cap in range(1, len(converted)):
            with (
                self.subTest(cap=cap),
                patch("src.tools.pylon_tools.ARTICLE_CONTENT_MAX_CHARS", cap),
            ):
                result = _format_article_content(html)
                shown, marker = result.split("\n\n[Article truncated:")
                self.assertLessEqual(len(shown), cap)
                self.assertEqual(shown.count("```") % 2, 0)
                self.assertEqual(
                    marker, f" showing {len(shown)} of {len(converted)} characters]"
                )

    @patch("src.tools.pylon_tools._fetch_collections", return_value={"General": "c1"})
    @patch("src.tools.pylon_tools._fetch_all_articles")
    def test_tool_returns_converted_content_and_metadata(
        self, mock_articles, mock_collections
    ):
        for length in (20, ARTICLE_CONTENT_MAX_CHARS + 100):
            with self.subTest(length=length):
                html = f"<p>{'x' * length}</p>"
                mock_articles.return_value = [
                    {
                        "id": "a1",
                        "title": "Setup",
                        "collection_id": "c1",
                        "identifier": "123",
                        "slug": "setup",
                        "current_published_content_html": html,
                    }
                ]
                result = get_support_article_content.invoke({"article_id": "a1"})
                self.assertEqual(
                    result,
                    "ID: a1\nTitle: Setup\nURL: https://support.langchain.com/articles/123-setup\n"
                    f"Collection: General\n\nContent:\n{_format_article_content(html)}",
                )
