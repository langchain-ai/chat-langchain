"""Tests for the check_links tool — verifies it works in both sync and async contexts.

The core bug: `asyncio.run()` raises `RuntimeError: This event loop is already running`
when called from within an async context (e.g. a LangGraph agent node). The fix is to
make `check_links` itself `async` so it can simply `await _check_urls_async(...)`.

Test strategy: use `unittest.mock` to patch the internal HTTP layer so the tests are
fast, deterministic, and require no real network access or LangSmith credentials.
"""

import socket
from unittest.mock import patch

import httpx
import pytest

# ---------------------------------------------------------------------------
# Helpers to build fake LinkCheckResult objects without importing the whole
# module (which would trigger import-time side effects).
# ---------------------------------------------------------------------------
from src.tools.link_check_tools import (
    LinkCheckResult,
    _cache,
    _check_single_url,
    _format_results,
    check_links,
)


@pytest.fixture(autouse=True)
def public_dns(monkeypatch):
    _cache.clear()
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))
        ],
    )
    yield
    _cache.clear()


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8000/ok",
        "http://LOCALHOST./metrics",
        "http://localhost:8000/docs/../../metrics",
        "http://app.localhost/",
        "http://127.0.0.1/",
        "http://0.0.0.0/",
        "http://[::1]/",
        "http://[::]/",
        "http://[::ffff:127.0.0.1]/",
        "http://metadata.google.internal/computeMetadata/v1/",
        "http://redis/",
        "http://10.0.0.1/",
        "http://169.254.169.254/",
        "http://224.0.0.1/",
        "http://240.0.0.1/",
        "http://[fc00::1]/",
        "http://[fe80::1]/",
        "http://[ff02::1]/",
        "http://[100::1]/",
        "ftp://example.com/file",
        "http:///missing-host",
        "http://example.com:invalid/",
    ],
)
async def test_blocked_destinations_make_no_http_requests(url):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await _check_single_url(client, url, timeout=1.0)

    assert result == LinkCheckResult(url=url, valid=False, error="Host not allowed")
    assert requests == []
    assert url not in _cache


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "addresses",
    [
        [],
        ["127.0.0.1"],
        ["10.0.0.1"],
        ["169.254.169.254"],
        ["0.0.0.0"],
        ["224.0.0.1"],
        ["240.0.0.1"],
        ["::1"],
        ["fc00::1"],
        ["fe80::1"],
        ["ff02::1"],
        ["100::1"],
        ["::"],
        ["::ffff:127.0.0.1"],
        ["8.8.8.8", "10.0.0.1"],
        ["8.8.8.8", "::1"],
    ],
)
async def test_dns_must_resolve_only_to_public_addresses(monkeypatch, addresses):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (
                socket.AF_INET6 if ":" in address else socket.AF_INET,
                socket.SOCK_STREAM,
                6,
                "",
                (address, 443),
            )
            for address in addresses
        ],
    )
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200)

    url = "https://example.com/"
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await _check_single_url(client, url, timeout=1.0)

    assert result.error == "Host not allowed"
    assert not result.valid
    assert requests == []
    assert url not in _cache


@pytest.mark.asyncio
async def test_dns_failure_is_generic_and_makes_no_http_request(monkeypatch):
    def fail_resolution(*args, **kwargs):
        raise socket.gaierror("sensitive DNS details")

    monkeypatch.setattr(socket, "getaddrinfo", fail_resolution)
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await _check_single_url(client, "http://example.com/", timeout=1.0)

    assert result.error == "Host not allowed"
    assert requests == []
    assert "sensitive" not in _format_results([result])


@pytest.mark.asyncio
@pytest.mark.parametrize("cached_redirect", [False, True])
async def test_cache_cannot_bypass_destination_guard(cached_redirect):
    url = "http://example.com/" if cached_redirect else "http://localhost/"
    _cache[url] = LinkCheckResult(
        url=url,
        valid=True,
        status_code=200,
        final_url="http://localhost/" if cached_redirect else None,
    )
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await _check_single_url(client, url, timeout=1.0)

    assert result == LinkCheckResult(url=url, valid=False, error="Host not allowed")
    assert requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["head", "stream", "fallback"])
@pytest.mark.parametrize(
    "destination",
    [
        "http://127.0.0.1/metrics",
        "http://[::1]/",
        "//metadata.google.internal/",
        "http://redis/",
    ],
)
async def test_redirect_to_internal_destination_is_not_requested(path, destination):
    url = (
        "https://docs.langchain.com/start"
        if path == "stream"
        else "http://example.com/start"
    )
    requests = []

    def respond(request):
        requests.append((request.method, str(request.url)))
        if path == "fallback" and request.method == "HEAD":
            return httpx.Response(405)
        return httpx.Response(302, headers={"Location": destination})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond), follow_redirects=True
    ) as client:
        result = await _check_single_url(client, url, timeout=1.0)

    expected_methods = (
        ["HEAD", "GET"]
        if path == "fallback"
        else ["GET" if path == "stream" else "HEAD"]
    )
    assert requests == [(method, url) for method in expected_methods]
    assert result == LinkCheckResult(url=url, valid=False, error="Host not allowed")
    assert url not in _cache


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,method",
    [
        ("http://example.com/start", "HEAD"),
        ("https://docs.langchain.com/start", "GET"),
        ("http://8.8.8.8/start", "HEAD"),
        ("https://[2606:4700:4700::1111]/start", "HEAD"),
    ],
)
async def test_public_links_and_relative_redirects_are_checked(url, method):
    requests = []

    def respond(request):
        requests.append((request.method, str(request.url)))
        if request.url.path == "/start":
            return httpx.Response(302, headers={"Location": "/guide"})
        return httpx.Response(200, text="<title>Documentation</title>")

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await _check_single_url(client, url, timeout=1.0)

    final_url = url.replace("/start", "/guide")
    assert requests == [(method, url), (method, final_url)]
    assert result == LinkCheckResult(
        url=url, valid=True, status_code=200, final_url=final_url
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [200, 404])
async def test_public_head_fallback_preserves_http_status(status_code):
    requests = []

    def respond(request):
        requests.append(request.method)
        return httpx.Response(405 if request.method == "HEAD" else status_code)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await _check_single_url(client, "http://example.com/", timeout=1.0)

    assert requests == ["HEAD", "GET"]
    assert result.status_code == status_code
    assert result.valid == (status_code == 200)
    assert result.error == (None if status_code == 200 else "HTTP 404")


@pytest.mark.asyncio
async def test_head_fallback_rechecks_dns(monkeypatch):
    requests = []

    def respond(request):
        requests.append(request.method)
        monkeypatch.setattr(
            socket,
            "getaddrinfo",
            lambda *args, **kwargs: [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))
            ],
        )
        return httpx.Response(405)

    url = "http://example.com/"
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await _check_single_url(client, url, timeout=1.0)

    assert requests == ["HEAD"]
    assert result == LinkCheckResult(url=url, valid=False, error="Host not allowed")
    assert url not in _cache


@pytest.mark.asyncio
async def test_public_ipv6_dns_result_is_allowed(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (
                socket.AF_INET6,
                socket.SOCK_STREAM,
                6,
                "",
                ("2606:4700:4700::1111", 443, 0, 0),
            )
        ],
    )
    requests = []

    def respond(request):
        requests.append(str(request.url))
        return httpx.Response(200)

    url = "https://example.com/"
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await _check_single_url(client, url, timeout=1.0)

    assert requests == [url]
    assert result.valid


@pytest.mark.asyncio
@pytest.mark.parametrize("hops", [5, 6])
async def test_redirects_are_limited_to_five_hops(hops):
    requests = []

    def respond(request):
        requests.append(str(request.url))
        step = int(request.url.path.strip("/"))
        if step < hops:
            return httpx.Response(302, headers={"Location": f"/{step + 1}"})
        return httpx.Response(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await _check_single_url(client, "https://example.com/0", timeout=1.0)

    assert len(requests) == 6
    assert result.valid == (hops == 5)
    assert result.error == (None if hops == 5 else "Too many redirects")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure,error",
    [
        (httpx.ConnectError("sensitive connection details"), "Connection failed"),
        (httpx.ReadTimeout("sensitive timeout details"), "Request timed out"),
        (RuntimeError("sensitive unexpected details"), "Request failed"),
    ],
)
async def test_network_failures_do_not_expose_exception_details(failure, error):
    def respond(request):
        raise failure

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await _check_single_url(client, "https://example.com/", timeout=1.0)

    assert not result.valid
    assert result.error == error
    assert "sensitive" not in _format_results([result])


def test_format_results_suppresses_raw_errors():
    results = [
        LinkCheckResult(url="https://example.com/", valid=False, error=error)
        for error in [
            "Connection failed: sensitive details",
            "DNS: sensitive details",
            "Error: sensitive details",
        ]
    ]

    formatted = _format_results(results)

    assert "sensitive" not in formatted
    assert "Request failed" in formatted
