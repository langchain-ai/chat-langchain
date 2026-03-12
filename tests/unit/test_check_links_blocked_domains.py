"""Tests that check_links rejects banned/deprecated domains before making HTTP requests.

Root cause: python.langchain.com and js.langchain.com redirect to docs.langchain.com but
the pages that exist there are not the correct destination pages — the agent was getting a
false "valid" signal from check_links because those domains return HTTP 200 (or a redirect
that resolves to a 200). The fix adds a BLOCKED_DOMAINS constant and short-circuits the
HTTP check so those domains are always reported as invalid with an actionable error message.

Test strategy:
- Tests of _check_single_url use asyncio.run() (no pytest-asyncio available).
- Tests of the check_links tool use asyncio.run(check_links.ainvoke(...)) since the tool
  does not support sync invocation (StructuredTool wrapping an async def).
- HTTP calls are mocked via a fake httpx.AsyncClient to confirm no network access occurs
  for blocked domains.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from src.tools.link_check_tools import (
    BLOCKED_DOMAINS,
    LinkCheckResult,
    _check_single_url,
    _check_urls_async,
    check_links,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_async_check_mock(results: list[LinkCheckResult]):
    """Return an async function that ignores its arguments and returns *results*."""

    async def _mock(urls, timeout):  # noqa: ARG001
        return results

    return _mock


def _make_fake_client() -> MagicMock:
    """Return a mock httpx.AsyncClient that must never be called for blocked domains."""
    client = MagicMock()
    client.head = AsyncMock(
        side_effect=AssertionError("HTTP call made for blocked domain")
    )
    client.get = AsyncMock(
        side_effect=AssertionError("HTTP call made for blocked domain")
    )
    client.stream = MagicMock(
        side_effect=AssertionError("HTTP call made for blocked domain")
    )
    return client


# ===========================================================================
# 1. Verify BLOCKED_DOMAINS constant is defined with the right contents
# ===========================================================================


def test_blocked_domains_constant_exists():
    """BLOCKED_DOMAINS must be defined and contain the two deprecated domains."""
    assert "python.langchain.com" in BLOCKED_DOMAINS
    assert "js.langchain.com" in BLOCKED_DOMAINS


def test_docs_langchain_com_not_in_blocked_domains():
    """docs.langchain.com is the canonical domain and must NOT be blocked."""
    assert "docs.langchain.com" not in BLOCKED_DOMAINS


# ===========================================================================
# 2. _check_single_url rejects blocked domains without making HTTP calls
# ===========================================================================


def test_python_langchain_com_is_blocked():
    """python.langchain.com URL must be returned as invalid without any HTTP request."""
    url = "https://python.langchain.com/api_reference/openai/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html"
    client = _make_fake_client()

    result = asyncio.run(_check_single_url(client, url, timeout=10.0))

    assert result.valid is False
    assert result.url == url
    assert result.error is not None
    assert "Deprecated domain" in result.error
    assert "docs.langchain.com" in result.error
    # No AssertionError means no HTTP call was attempted.


def test_js_langchain_com_is_blocked():
    """js.langchain.com URL must be returned as invalid without any HTTP request."""
    url = "https://js.langchain.com/docs/modules/model_io/chat/"
    client = _make_fake_client()

    result = asyncio.run(_check_single_url(client, url, timeout=10.0))

    assert result.valid is False
    assert result.url == url
    assert result.error is not None
    assert "Deprecated domain" in result.error
    assert "docs.langchain.com" in result.error


def test_blocked_domain_result_is_not_marked_valid():
    """Blocked domain results must have valid=False, not True."""
    url = "https://python.langchain.com/docs/introduction/"
    client = _make_fake_client()

    result = asyncio.run(_check_single_url(client, url, timeout=10.0))

    assert result.valid is False


# ===========================================================================
# 3. Regular and approved domains still work normally (not blocked)
# ===========================================================================


def test_docs_langchain_com_passes_through_to_http():
    """docs.langchain.com must not be blocked — the function must attempt HTTP."""
    url = "https://docs.langchain.com/docs/introduction/"

    # Provide a realistic mock that returns a valid streamed response
    client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.url = url

    # docs.langchain.com triggers soft-404 content check (stream)
    async def _fake_aiter_text():
        yield "<html><title>Introduction</title></html>"

    mock_response.aiter_text = _fake_aiter_text

    mock_context = MagicMock()
    mock_context.__aenter__ = AsyncMock(return_value=mock_response)
    mock_context.__aexit__ = AsyncMock(return_value=False)
    client.stream = MagicMock(return_value=mock_context)

    result = asyncio.run(_check_single_url(client, url, timeout=10.0))

    # docs.langchain.com is not blocked — no "Deprecated domain" error
    assert result.error != "Deprecated domain: use docs.langchain.com instead"
    # HTTP was attempted (stream was called)
    client.stream.assert_called_once()


def test_regular_external_url_is_not_blocked():
    """External URLs like example.com must still be checked normally (not blocked)."""
    url = "https://example.com"
    client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.url = url
    client.head = AsyncMock(return_value=mock_response)

    result = asyncio.run(_check_single_url(client, url, timeout=10.0))

    # Not blocked — no "Deprecated domain" error
    assert result.error != "Deprecated domain: use docs.langchain.com instead"
    assert result.valid is True
    client.head.assert_called_once()


# ===========================================================================
# 4. check_links tool integration: blocked domains show up as invalid
# ===========================================================================


def test_check_links_reports_python_langchain_com_invalid():
    """check_links must report python.langchain.com as invalid."""
    fake_results = [
        LinkCheckResult(
            url="https://python.langchain.com/api_reference/openai/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html",
            valid=False,
            error="Deprecated domain: use docs.langchain.com instead",
        ),
    ]

    with patch(
        "src.tools.link_check_tools._check_urls_async",
        new=_make_async_check_mock(fake_results),
    ):
        result = asyncio.run(
            check_links.ainvoke(
                {
                    "urls": [
                        "https://python.langchain.com/api_reference/openai/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html"
                    ]
                }
            )
        )

    assert "0/1 valid" in result
    assert "Deprecated domain" in result
    assert "docs.langchain.com" in result


def test_check_links_reports_js_langchain_com_invalid():
    """check_links must report js.langchain.com as invalid."""
    fake_results = [
        LinkCheckResult(
            url="https://js.langchain.com/docs/modules/model_io/",
            valid=False,
            error="Deprecated domain: use docs.langchain.com instead",
        ),
    ]

    with patch(
        "src.tools.link_check_tools._check_urls_async",
        new=_make_async_check_mock(fake_results),
    ):
        result = asyncio.run(
            check_links.ainvoke(
                {"urls": ["https://js.langchain.com/docs/modules/model_io/"]}
            )
        )

    assert "0/1 valid" in result
    assert "Deprecated domain" in result


def test_check_links_mixed_blocked_and_valid():
    """Mixed list: blocked domain invalid, docs.langchain.com valid."""
    fake_results = [
        LinkCheckResult(
            url="https://python.langchain.com/some/path",
            valid=False,
            error="Deprecated domain: use docs.langchain.com instead",
        ),
        LinkCheckResult(
            url="https://docs.langchain.com/some/path",
            valid=True,
            status_code=200,
        ),
    ]

    with patch(
        "src.tools.link_check_tools._check_urls_async",
        new=_make_async_check_mock(fake_results),
    ):
        result = asyncio.run(
            check_links.ainvoke(
                {
                    "urls": [
                        "https://python.langchain.com/some/path",
                        "https://docs.langchain.com/some/path",
                    ]
                }
            )
        )

    assert "1/2 valid" in result
    assert "Deprecated domain" in result
    assert "docs.langchain.com/some/path" in result


# ===========================================================================
# 5. End-to-end: _check_urls_async calls _check_single_url which blocks domains
#    (verifies the full async pipeline without mocking _check_urls_async)
# ===========================================================================


def test_check_urls_async_blocks_python_langchain_com():
    """_check_urls_async must return an invalid result for python.langchain.com."""
    url = "https://python.langchain.com/some/api/path"

    async def _run():
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.head = AsyncMock(
                side_effect=AssertionError("Should not reach HTTP for blocked domain")
            )
            mock_client.get = AsyncMock(
                side_effect=AssertionError("Should not reach HTTP for blocked domain")
            )
            mock_client.stream = MagicMock(
                side_effect=AssertionError("Should not reach HTTP for blocked domain")
            )
            mock_client_cls.return_value = mock_client

            results = await _check_urls_async([url], timeout=10.0)
            return results

    results = asyncio.run(_run())
    assert len(results) == 1
    assert results[0].valid is False
    assert "Deprecated domain" in results[0].error
