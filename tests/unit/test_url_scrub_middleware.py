"""Tests for the URL scrub middleware.

The docs agent occasionally fabricates `https://docs.langchain.com/docs/...`
URLs from its training-data memory of the legacy docs URL scheme. The
production docs site no longer hosts content under `/docs/...`; the canonical
path prefixes are `/oss/python/...`, `/oss/javascript/...`, `/langsmith/...`,
etc. `UrlScrubMiddleware` strips fabricated `/docs/...` URLs from the final
assistant message as a defense-in-depth guard.
"""

from src.middleware.url_scrub_middleware import (
    _REPLACEMENT_MARKER,
    scrub_fabricated_docs_urls,
)


class TestScrubFabricatedDocsUrls:
    def test_removes_fabricated_legacy_docs_link(self) -> None:
        text = (
            "See [Tool Calling](https://docs.langchain.com/docs/concepts/tool_calling) "
            "for details."
        )
        scrubbed, replaced = scrub_fabricated_docs_urls(text)
        assert replaced == 1
        assert "docs.langchain.com/docs/concepts/tool_calling" not in scrubbed
        assert _REPLACEMENT_MARKER in scrubbed

    def test_preserves_valid_oss_python_link(self) -> None:
        text = (
            "See [Streaming](https://docs.langchain.com/oss/python/langgraph/streaming) "
            "for details."
        )
        scrubbed, replaced = scrub_fabricated_docs_urls(text)
        assert replaced == 0
        assert (
            "https://docs.langchain.com/oss/python/langgraph/streaming" in scrubbed
        )

    def test_mixed_response_only_scrubs_fabricated_links(self) -> None:
        text = (
            "**Relevant docs:**\n\n"
            "- [Streaming](https://docs.langchain.com/oss/python/langgraph/streaming)\n"
            "- [Tool Calling](https://docs.langchain.com/docs/concepts/tool_calling)\n"
            "- [LangSmith](https://docs.langchain.com/langsmith/observability)\n"
        )
        scrubbed, replaced = scrub_fabricated_docs_urls(text)
        assert replaced == 1
        # Legit URLs preserved
        assert (
            "https://docs.langchain.com/oss/python/langgraph/streaming" in scrubbed
        )
        assert "https://docs.langchain.com/langsmith/observability" in scrubbed
        # Fabricated URL removed
        assert "docs.langchain.com/docs/concepts/tool_calling" not in scrubbed

    def test_scrubs_bare_fabricated_url(self) -> None:
        text = "See https://docs.langchain.com/docs/get-started/introduction for more."
        scrubbed, replaced = scrub_fabricated_docs_urls(text)
        assert replaced == 1
        assert "docs.langchain.com/docs/get-started" not in scrubbed

    def test_returns_zero_when_no_fabricated_urls(self) -> None:
        text = (
            "All good — see "
            "[LangGraph Platform](https://docs.langchain.com/langgraph-platform/overview)."
        )
        scrubbed, replaced = scrub_fabricated_docs_urls(text)
        assert replaced == 0
        assert scrubbed == text
