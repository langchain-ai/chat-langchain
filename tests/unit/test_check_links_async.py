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

# ---------------------------------------------------------------------------
# Helpers to build fake LinkCheckResult objects without importing the whole
# module (which would trigger import-time side effects).
# ---------------------------------------------------------------------------
from src.tools.link_check_tools import (
    MAX_REDIRECTS,
    LinkCheckResult,
    _cache,
    _check_single_url,
    _check_urls_async,
    _format_results,
    _is_allowed_host,
    check_links,
)


@pytest.fixture(autouse=True)
def public_dns():
    _cache.clear()
    with patch(
        "asyncio.BaseEventLoop.getaddrinfo",
        new=AsyncMock(
            return_value=[
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
            ]
        ),
    ) as resolver:
        yield resolver
    _cache.clear()


@pytest.fixture
def http_requests(monkeypatch):
    requests = []
    responses = {}
    client_class = httpx.AsyncClient

    def handler(request):
        assert _is_allowed_host(str(request.url)), "Request to a disallowed host"
        requests.append(request)
        return httpx.Response(
            **responses.get(
                str(request.url), {"status_code": 200, "text": "Documentation"}
            )
        )

    def client_factory(**kwargs):
        assert kwargs["follow_redirects"] is False
        assert kwargs["trust_env"] is False
        return client_class(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr("src.tools.link_check_tools.httpx.AsyncClient", client_factory)
    return requests, responses


class _FakeStreamResponse:
    """Minimal async streaming response for _check_single_url tests."""

    def __init__(self, url: str, status_code: int, content: str):
        self.url = url
        self.status_code = status_code
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
        assert _is_allowed_host(url)
        assert kwargs["follow_redirects"] is False
        return _FakeStreamResponse(url, self.status_code, self.content)


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


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data/",
        "http://127.0.0.1:8080/",
        "http://metadata.google.internal/",
        "https://example.oast.site/",
        "http://[::1]/",
        "https://evil-langchain.com/",
        "https://langchain.com.attacker.net/",
        "https://github.com/other-org/repo",
        "https://github.com/langchain-ai/../other-org/repo",
        "https://github.com/langchain-ai/%2e%2e/other-org/repo",
        "https://github.com/langchain-ai/%5c../other-org/repo",
        "https://docs.langchain.com@example.oast.site/",
        "https://example.oast.site@docs.langchain.com/",
    ],
)
@pytest.mark.asyncio
async def test_disallowed_urls_are_not_requested(url, http_requests, public_dns):
    requests, _ = http_requests

    result = await check_links.ainvoke({"urls": [url]})

    assert "0/1 valid" in result
    assert "Not checked: only LangChain documentation hosts are allowed" in result
    assert requests == []
    public_dns.assert_not_awaited()
    assert not _cache[url].valid


@pytest.mark.parametrize(
    "url",
    [
        "https://docs.langchain.com/oss/python/langchain/overview",
        "https://DOCS.LANGCHAIN.COM/overview",
        "https://langchain.com/",
        "https://langsmith.com/",
        "https://docs.langsmith.com/",
        "https://github.com/langchain-ai/",
        "https://github.com/langchain-ai/langchain",
    ],
)
@pytest.mark.asyncio
async def test_allowed_urls_are_checked(url, http_requests, public_dns):
    requests, _ = http_requests

    results = await _check_urls_async([url], timeout=1.0)

    assert results[0].valid
    assert results[0].status_code == 200
    assert len(requests) == 1
    public_dns.assert_awaited_once()


@pytest.mark.parametrize(
    "target",
    [
        "https://example.oast.site/",
        "//169.254.169.254/latest/meta-data/",
        "http://127.0.0.1:8080/",
        "http://metadata.google.internal/",
        "https://github.com/other-org/repo",
        "file:///etc/passwd",
    ],
)
@pytest.mark.parametrize(
    "url", ["https://docs.langchain.com/start", "https://github.com/langchain-ai/start"]
)
@pytest.mark.asyncio
async def test_disallowed_redirect_is_not_followed(
    url, target, http_requests, public_dns
):
    requests, responses = http_requests
    responses[url] = {"status_code": 302, "headers": {"location": target}}

    results = await _check_urls_async([url], timeout=1.0)

    assert not results[0].valid
    assert results[0].final_url is None
    assert [str(request.url) for request in requests] == [url]
    public_dns.assert_awaited_once()
    assert target not in _format_results(results)


@pytest.mark.asyncio
async def test_relative_and_allowed_redirects_are_followed(http_requests, public_dns):
    requests, responses = http_requests
    url = "https://github.com/langchain-ai/start"
    intermediate = "https://github.com/langchain-ai/next"
    target = "https://docs.langchain.com/overview"
    responses[url] = {"status_code": 301, "headers": {"location": "next"}}
    responses[intermediate] = {"status_code": 308, "headers": {"location": target}}

    results = await _check_urls_async([url], timeout=1.0)

    assert results[0].valid
    assert results[0].final_url == target
    assert [str(request.url) for request in requests] == [url, intermediate, target]
    assert [request.method for request in requests] == ["HEAD", "HEAD", "GET"]
    assert public_dns.await_count == 3


@pytest.mark.asyncio
async def test_redirect_limit_is_enforced(http_requests):
    requests, responses = http_requests
    url = "https://docs.langchain.com/loop"
    responses[url] = {"status_code": 302, "headers": {"location": "/loop"}}

    results = await _check_urls_async([url], timeout=1.0)

    assert not results[0].valid
    assert results[0].error == "Too many redirects"
    assert len(requests) == MAX_REDIRECTS + 1


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.1",
        "169.254.169.254",
        "0.0.0.0",
        "224.0.0.1",
        "240.0.0.1",
        "::1",
        "fe80::1",
        "::",
        "ff02::1",
        "::ffff:127.0.0.1",
    ],
)
@pytest.mark.asyncio
async def test_non_public_dns_is_not_requested(address, http_requests, public_dns):
    requests, _ = http_requests
    public_dns.return_value.append(
        (socket.AF_INET6, socket.SOCK_STREAM, 6, "", (address, 443))
    )

    results = await _check_urls_async(
        ["https://docs.langchain.com/overview"], timeout=1.0
    )

    assert not results[0].valid
    assert "non-public address" in results[0].error
    assert requests == []


@pytest.mark.asyncio
async def test_redirect_to_non_public_dns_is_not_requested(http_requests, public_dns):
    requests, responses = http_requests
    url = "https://docs.langchain.com/start"
    responses[url] = {
        "status_code": 302,
        "headers": {"location": "https://support.langchain.com/article"},
    }
    public_dns.side_effect = [
        public_dns.return_value,
        [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))],
    ]

    results = await _check_urls_async([url], timeout=1.0)

    assert not results[0].valid
    assert [str(request.url) for request in requests] == [url]


@pytest.mark.asyncio
async def test_dns_failure_is_generic(http_requests, public_dns):
    requests, _ = http_requests
    public_dns.side_effect = socket.gaierror("internal resolver details")

    results = await _check_urls_async(
        ["https://docs.langchain.com/overview"], timeout=1.0
    )

    assert not results[0].valid
    assert "internal resolver details" not in _format_results(results)
    assert requests == []


@pytest.mark.asyncio
async def test_head_405_falls_back_to_get_without_redirects(public_dns):
    url = "https://github.com/langchain-ai/langchain"
    client = MagicMock()
    client.head = AsyncMock(return_value=httpx.Response(405))
    client.get = AsyncMock(
        return_value=httpx.Response(302, headers={"location": "http://127.0.0.1/"})
    )

    result = await _check_single_url(client, url, timeout=1.0)

    assert not result.valid
    client.head.assert_awaited_once_with(url, timeout=1.0, follow_redirects=False)
    client.get.assert_awaited_once_with(url, timeout=1.0, follow_redirects=False)


@pytest.mark.asyncio
async def test_redirected_soft_404_is_still_invalid(http_requests):
    requests, responses = http_requests
    url = "https://github.com/langchain-ai/start"
    target = "https://docs.langchain.com/missing"
    responses[url] = {"status_code": 302, "headers": {"location": target}}
    responses[target] = {"status_code": 200, "text": "<title>Page not found</title>"}

    results = await _check_urls_async([url], timeout=1.0)

    assert not results[0].valid
    assert results[0].error == "Soft 404: Page shows 'not found' content"
    assert len(requests) == 2


def test_format_results_hides_connection_details_and_disallowed_final_urls():
    results = [
        LinkCheckResult(
            url="https://docs.langchain.com/",
            valid=False,
            error="Connection failed: internal address",
        ),
        LinkCheckResult(
            url="https://langchain.com/",
            valid=True,
            final_url="http://127.0.0.1/private",
        ),
    ]

    output = _format_results(results)

    assert "internal address" not in output
    assert "127.0.0.1" not in output
    assert "Unable to check URL" in output


@pytest.mark.parametrize("url", ["not-a-url", "ftp://docs.langchain.com/page"])
@pytest.mark.asyncio
async def test_invalid_format_is_not_requested(url, http_requests, public_dns):
    requests, _ = http_requests

    results = await _check_urls_async([url], timeout=1.0)

    assert not results[0].valid
    assert results[0].error == "Invalid URL format"
    assert requests == []
    public_dns.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_dns_result_is_not_requested(http_requests, public_dns):
    requests, _ = http_requests
    public_dns.return_value = []

    results = await _check_urls_async(["https://docs.langchain.com/"], timeout=1.0)

    assert not results[0].valid
    assert requests == []


@pytest.mark.asyncio
async def test_cached_result_does_not_repeat_requests(http_requests, public_dns):
    requests, _ = http_requests
    url = "https://docs.langchain.com/overview"

    first = await _check_urls_async([url], timeout=1.0)
    second = await _check_urls_async([url], timeout=1.0)

    assert first == second
    assert first[0].valid
    assert len(requests) == 1
    public_dns.assert_awaited_once()


@pytest.mark.asyncio
async def test_connection_error_is_generic():
    client = MagicMock()
    client.head = AsyncMock(
        side_effect=httpx.ConnectError("private connection details")
    )

    result = await _check_single_url(
        client, "https://github.com/langchain-ai/", timeout=1.0
    )

    assert not result.valid
    assert result.error == "Connection failed"
    assert "private connection details" not in _format_results([result])
