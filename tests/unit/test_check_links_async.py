"""Tests for the check_links tool — verifies it works in both sync and async contexts.

The core bug: `asyncio.run()` raises `RuntimeError: This event loop is already running`
when called from within an async context (e.g. a LangGraph agent node). The fix is to
make `check_links` itself `async` so it can simply `await _check_urls_async(...)`.

Test strategy: use `unittest.mock` to patch the internal HTTP layer so the tests are
fast, deterministic, and require no real network access or LangSmith credentials.
"""

import socket
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.tools.link_check_tools import (
    MAX_REDIRECTS,
    LinkCheckResult,
    _cache,
    _check_single_url,
    _check_urls_async,
    check_links,
)


class _FakeStreamResponse:
    """Minimal async streaming response for _check_single_url tests."""

    def __init__(self, url: str, status_code: int, content: str):
        self.url = url
        self.status_code = status_code
        self.headers = {}
        self._content = content

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):  # noqa: ANN001
        return None

    async def aiter_text(self):
        yield self._content


class _FakeStreamingClient:
    """Minimal client that exercises the soft-404 streaming path."""

    def __init__(self, content: str, status_code: int = 200):
        self.content = content
        self.status_code = status_code

    def stream(self, method: str, url: str, **kwargs):  # noqa: ARG002
        return _FakeStreamResponse(url, self.status_code, self.content)


@pytest.fixture(autouse=True)
def public_dns():
    _cache.clear()
    with patch(
        "src.tools.link_check_tools.socket.getaddrinfo",
        return_value=[
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))
        ],
    ) as resolver:
        yield resolver
    _cache.clear()


@pytest.fixture
def mock_http():
    responses = {}
    requests = []
    client_class = httpx.AsyncClient

    def handle_request(request):
        requests.append(request)
        status, headers, content = responses[str(request.url)]
        return httpx.Response(status, headers=headers, text=content)

    def make_client(**kwargs):
        assert kwargs["follow_redirects"] is False
        return client_class(transport=httpx.MockTransport(handle_request), **kwargs)

    with patch("src.tools.link_check_tools.httpx.AsyncClient", side_effect=make_client):
        yield responses, requests


# ---------------------------------------------------------------------------
# Fixture: a canned async replacement for _check_urls_async
# ---------------------------------------------------------------------------


def _make_async_check_mock(results: list[LinkCheckResult]):
    """Return an async function that ignores its arguments and returns *results*."""

    async def _mock_check_urls_async(urls, timeout):  # noqa: ARG001
        return results

    return _mock_check_urls_async


# ===========================================================================
# 1. Test that check_links works when called from within an async context
#    (this is the regression test — it FAILS before the fix because the old
#    code calls asyncio.run() inside an already-running event loop).
# ===========================================================================


@pytest.mark.asyncio
async def test_check_links_works_in_async_context():
    """check_links must be awaitable and must NOT raise RuntimeError in async context.

    Before the fix, calling `check_links.invoke(...)` from inside a coroutine would
    ultimately hit `asyncio.run()` which raises:
        RuntimeError: This event loop is already running
    After the fix, check_links is `async def` so it can be awaited directly.
    """
    fake_results = [
        LinkCheckResult(url="https://example.com", valid=True, status_code=200),
    ]

    with patch(
        "src.tools.link_check_tools._check_urls_async",
        new=_make_async_check_mock(fake_results),
    ):
        # After the fix check_links is a coroutine function (async def) wrapped
        # by @tool.  LangChain's StructuredTool exposes it as .acall / .ainvoke.
        # We call the underlying coroutine function directly to keep the test
        # independent of LangChain's tool-wrapping internals.
        result = await check_links.acall({"urls": ["https://example.com"]})

    assert "1/1 valid" in result
    assert "  - https://example.com" in result


@pytest.mark.asyncio
async def test_check_links_async_context_does_not_raise_runtime_error():
    """Explicitly assert that no RuntimeError is raised in an async context."""
    fake_results = [
        LinkCheckResult(url="https://docs.langchain.com/", valid=True, status_code=200),
    ]

    with patch(
        "src.tools.link_check_tools._check_urls_async",
        new=_make_async_check_mock(fake_results),
    ):
        raised = False
        try:
            await check_links.acall({"urls": ["https://docs.langchain.com/"]})
        except RuntimeError as exc:
            if "event loop is already running" in str(exc):
                raised = True

    assert not raised, (
        "RuntimeError('This event loop is already running') was raised — "
        "the fix has not been applied correctly."
    )


# ===========================================================================
# 2. Test that check_links works normally in a sync context
#    (the tool must still be usable when there is no running event loop,
#     e.g. in plain scripts or sync test runners).
# ===========================================================================


def test_check_links_works_in_sync_context():
    """check_links.invoke() must work when called outside an async context."""
    fake_results = [
        LinkCheckResult(url="https://example.com", valid=True, status_code=200),
        LinkCheckResult(
            url="https://bad.example.com",
            valid=False,
            status_code=404,
            error="HTTP 404",
        ),
    ]

    with patch(
        "src.tools.link_check_tools._check_urls_async",
        new=_make_async_check_mock(fake_results),
    ):
        result = check_links.invoke(
            {"urls": ["https://example.com", "https://bad.example.com"]}
        )

    assert "1/2 valid" in result
    assert "  - https://example.com" in result
    assert "  - https://bad.example.com:" in result
    assert "HTTP 404" in result


def test_check_links_sync_deduplicates_urls():
    """Duplicate URLs should only be checked once."""
    seen_urls: list[list[str]] = []

    async def _recording_mock(urls, timeout):  # noqa: ARG001
        seen_urls.append(list(urls))
        return [LinkCheckResult(url=u, valid=True, status_code=200) for u in urls]

    with patch(
        "src.tools.link_check_tools._check_urls_async",
        new=_recording_mock,
    ):
        result = check_links.invoke(
            {
                "urls": [
                    "https://example.com",
                    "https://example.com",
                    "https://example.com",
                ]
            }
        )

    # _check_urls_async should have been called with exactly ONE unique URL
    assert len(seen_urls) == 1
    assert seen_urls[0] == ["https://example.com"]
    assert "1/1 valid" in result


# ===========================================================================
# 3. Edge cases
# ===========================================================================


def test_check_links_empty_list():
    """Passing an empty list should return the 'no URLs' message without error."""
    # No mock needed — the early-return branch is hit before any async work.
    result = check_links.invoke({"urls": []})
    assert result == "No URLs provided to check."


@pytest.mark.asyncio
async def test_check_links_async_empty_list():
    """Empty list in async context should also return the 'no URLs' message."""
    result = await check_links.acall({"urls": []})
    assert result == "No URLs provided to check."


def test_check_links_invalid_url_format():
    """An invalid URL (not http/https) must be reported as invalid."""
    # We do NOT mock here — _check_single_url validates format synchronously via
    # _is_valid_url before making any network calls, so no HTTP is performed.
    # We still mock _check_urls_async to avoid needing a real event loop in the
    # sync path and to keep the test fast.
    fake_results = [
        LinkCheckResult(url="not-a-url", valid=False, error="Invalid URL format"),
    ]

    with patch(
        "src.tools.link_check_tools._check_urls_async",
        new=_make_async_check_mock(fake_results),
    ):
        result = check_links.invoke({"urls": ["not-a-url"]})

    assert "0/1 valid" in result
    assert "Invalid URL format" in result


@pytest.mark.asyncio
async def test_check_links_async_invalid_url_format():
    """Invalid URL in async context should also be reported correctly."""
    fake_results = [
        LinkCheckResult(
            url="ftp://not-supported", valid=False, error="Invalid URL format"
        ),
    ]

    with patch(
        "src.tools.link_check_tools._check_urls_async",
        new=_make_async_check_mock(fake_results),
    ):
        result = await check_links.acall({"urls": ["ftp://not-supported"]})

    assert "0/1 valid" in result
    assert "Invalid URL format" in result


@pytest.mark.asyncio
async def test_support_article_not_found_is_invalid_soft_404():
    """Support articles can return HTTP 200 while rendering an article-not-found page."""
    result = await _check_single_url(
        _FakeStreamingClient("<html><body><div>Article Not Found</div></body></html>"),
        "https://support.langchain.com/hc/en-us/articles/missing-article",
        timeout=1.0,
    )

    assert not result.valid
    assert result.status_code == 200
    assert result.error == "Soft 404: Page shows 'not found' content"


@pytest.mark.asyncio
async def test_support_article_normal_content_is_valid():
    """Normal support article content should still be treated as valid."""
    result = await _check_single_url(
        _FakeStreamingClient(
            "<html><body><div>Useful support article</div></body></html>"
        ),
        "https://support.langchain.com/hc/en-us/articles/existing-article",
        timeout=1.0,
    )

    assert result.valid
    assert result.status_code == 200
    assert result.error is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8080/",
        "http://169.254.169.254/latest/meta-data/",
        "http://metadata.google.internal/",
        "https://webhook.site/x",
        "http://[::1]/",
        "https://docs.langchain.com.evil.example/",
        "https://docs.langchain.com@evil.example/",
        "https://user:password@docs.langchain.com/",
        "https://github.com/other-org/repo",
        "https://github.com/langchain-ai/../other-org/repo",
        "https://github.com/langchain-ai/%2e%2e/other-org/repo",
    ],
)
async def test_disallowed_hosts_are_not_fetched_or_cached(url, mock_http, public_dns):
    responses, requests = mock_http
    result = (await _check_urls_async([url], 1.0))[0]

    assert not result.valid
    assert result.error == "Host not allowed for link checking"
    assert requests == []
    public_dns.assert_not_called()
    assert url not in _cache


@pytest.mark.asyncio
async def test_disallowed_host_cannot_use_cached_result(mock_http, public_dns):
    url = "http://127.0.0.1/"
    _cache[url] = LinkCheckResult(url=url, valid=True, status_code=200)
    result = (await _check_urls_async([url], 1.0))[0]

    assert not result.valid
    assert result.error == "Host not allowed for link checking"
    assert mock_http[1] == []
    public_dns.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.1",
        "169.254.169.254",
        "240.0.0.1",
        "224.0.0.1",
        "0.0.0.0",
        "100.64.0.1",
        "::1",
        "fc00::1",
        "fe80::1",
        "ff02::1",
        "::",
        "::ffff:127.0.0.1",
    ],
)
async def test_allowed_host_with_nonpublic_dns_is_not_fetched(
    address, mock_http, public_dns
):
    public_dns.return_value.append(
        (
            socket.AF_INET6 if ":" in address else socket.AF_INET,
            socket.SOCK_STREAM,
            6,
            "",
            (address, 0),
        )
    )
    url = "https://docs.langchain.com/"
    result = (await _check_urls_async([url], 1.0))[0]

    assert not result.valid
    assert result.error == "Host does not resolve to a public IP"
    assert mock_http[1] == []
    assert url not in _cache


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [[], socket.gaierror("DNS failed")])
async def test_dns_failure_is_not_fetched(failure, mock_http, public_dns):
    if isinstance(failure, Exception):
        public_dns.side_effect = failure
    else:
        public_dns.return_value = failure
    result = (await _check_urls_async(["https://docs.langchain.com/"], 1.0))[0]

    assert not result.valid
    assert result.error == "Host does not resolve to a public IP"
    assert mock_http[1] == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "https://docs.langchain.com/",
        "https://python.langchain.com/",
        "https://js.langchain.com/",
        "https://support.langchain.com/",
        "https://reference.langchain.com/",
        "https://smith.langchain.com/",
        "https://github.com/langchain-ai/langchain",
    ],
)
async def test_allowed_documentation_is_checked(url, mock_http):
    responses, requests = mock_http
    responses[url] = (200, {}, "<title>Documentation</title>")
    result = (await _check_urls_async([url], 1.0))[0]

    assert result.valid
    assert result.status_code == 200
    assert len(requests) == 1
    assert str(requests[0].url) == url


@pytest.mark.asyncio
async def test_tool_reports_disallowed_links_without_fetching(mock_http):
    responses, requests = mock_http
    url = "https://docs.langchain.com/"
    responses[url] = (200, {}, "<title>Documentation</title>")
    result = await check_links.ainvoke(
        {
            "urls": [
                url,
                "http://127.0.0.1:8080/",
                "http://169.254.169.254/latest/meta-data/",
                "http://metadata.google.internal/",
                "https://webhook.site/x",
            ]
        }
    )

    assert "1/5 valid" in result
    assert result.count("Host not allowed for link checking") == 4
    assert [str(request.url) for request in requests] == [url]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
@pytest.mark.parametrize(
    "destination",
    [
        "http://127.0.0.1/",
        "http://169.254.169.254/latest/meta-data/",
        "https://webhook.site/x",
        "https://github.com/other-org/repo",
    ],
)
async def test_redirect_to_disallowed_host_is_not_followed(
    status, destination, mock_http
):
    responses, requests = mock_http
    url = "https://docs.langchain.com/"
    responses[url] = (status, {"Location": destination}, "")
    result = (await _check_urls_async([url], 1.0))[0]

    assert not result.valid
    assert result.error == "Host not allowed for link checking"
    assert [str(request.url) for request in requests] == [url]
    assert url not in _cache


@pytest.mark.asyncio
async def test_redirect_to_allowed_host_rechecks_dns(mock_http, public_dns):
    responses, requests = mock_http
    url = "https://docs.langchain.com/"
    responses[url] = (302, {"Location": "https://reference.langchain.com/"}, "")
    public_dns.side_effect = [
        public_dns.return_value,
        [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))],
    ]
    result = (await _check_urls_async([url], 1.0))[0]

    assert not result.valid
    assert result.error == "Host does not resolve to a public IP"
    assert [str(request.url) for request in requests] == [url]
    assert public_dns.call_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content, valid",
    [
        ("<title>Documentation</title>", True),
        ("<title>Page not found</title>", False),
    ],
)
async def test_relative_redirect_preserves_soft_404_check(content, valid, mock_http):
    responses, requests = mock_http
    url = "https://docs.langchain.com/start"
    destination = "https://docs.langchain.com/guide"
    responses[url] = (302, {"Location": "/guide"}, "")
    responses[destination] = (200, {}, content)
    result = (await _check_urls_async([url], 1.0))[0]

    assert result.valid is valid
    assert result.final_url == destination
    assert [str(request.url) for request in requests] == [url, destination]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [200, 302])
async def test_head_to_get_fallback_validates_redirect(status):
    url = "https://github.com/langchain-ai/langchain"
    client = MagicMock()
    client.head = AsyncMock(
        return_value=httpx.Response(405, request=httpx.Request("HEAD", url))
    )
    client.get = AsyncMock(
        return_value=httpx.Response(
            status,
            headers={"Location": "http://127.0.0.1/"},
            request=httpx.Request("GET", url),
        )
    )
    result = await _check_single_url(client, url, 1.0)

    assert result.valid is (status == 200)
    assert result.error == (
        None if status == 200 else "Host not allowed for link checking"
    )
    client.head.assert_awaited_once_with(url, timeout=1.0)
    client.get.assert_awaited_once_with(url, timeout=1.0)


@pytest.mark.asyncio
async def test_redirect_limit(mock_http):
    responses, requests = mock_http
    url = "https://docs.langchain.com/"
    responses[url] = (302, {"Location": url}, "")
    result = (await _check_urls_async([url], 1.0))[0]

    assert not result.valid
    assert result.error == "Too many redirects"
    assert len(requests) == MAX_REDIRECTS + 1
