"""Tests for the check_links tool — verifies it works in both sync and async contexts.

The core bug: `asyncio.run()` raises `RuntimeError: This event loop is already running`
when called from within an async context (e.g. a LangGraph agent node). The fix is to
make `check_links` itself `async` so it can simply `await _check_urls_async(...)`.

Test strategy: use `unittest.mock` to patch the internal HTTP layer so the tests are
fast, deterministic, and require no real network access or LangSmith credentials.
"""

import asyncio
import socket
import ssl
from unittest.mock import AsyncMock, patch

import httpcore
import httpx
import pytest

# ---------------------------------------------------------------------------
# Helpers to build fake LinkCheckResult objects without importing the whole
# module (which would trigger import-time side effects).
# ---------------------------------------------------------------------------
from src.tools.link_check_tools import (
    LinkCheckResult,
    _check_single_url,
    _check_urls_async,
    check_links,
)
from src.tools.link_check_tools import _cache as link_cache


@pytest.fixture(autouse=True)
def public_dns(monkeypatch):
    link_cache.clear()
    resolver = AsyncMock(
        return_value=[
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
        ]
    )
    monkeypatch.setattr(asyncio.BaseEventLoop, "getaddrinfo", resolver)
    yield resolver
    link_cache.clear()


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
    # We still mock _check_urls_async to avoid needing a real event loop in the
    # sync path and to keep the test fast.
    fake_results = [
        LinkCheckResult(url="not-a-url", valid=False, error="URL not allowed"),
    ]

    with patch(
        "src.tools.link_check_tools._check_urls_async",
        new=_make_async_check_mock(fake_results),
    ):
        result = check_links.invoke({"urls": ["not-a-url"]})

    assert "0/1 valid" in result
    assert "URL not allowed" in result


@pytest.mark.asyncio
async def test_check_links_async_invalid_url_format():
    """Invalid URL in async context should also be reported correctly."""
    fake_results = [
        LinkCheckResult(
            url="ftp://not-supported", valid=False, error="URL not allowed"
        ),
    ]

    with patch(
        "src.tools.link_check_tools._check_urls_async",
        new=_make_async_check_mock(fake_results),
    ):
        result = await check_links.acall({"urls": ["ftp://not-supported"]})

    assert "0/1 valid" in result
    assert "URL not allowed" in result


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
        "http://localhost:8000/metrics",
        "http://localhost./",
        "http://app.localhost/",
        "http://127.0.0.1/",
        "http://0.0.0.0/",
        "http://[::1]/",
        "http://[::]/",
        "http://10.0.0.1/",
        "http://172.16.0.1/",
        "http://192.168.1.1/",
        "http://169.254.169.254/",
        "http://[fe80::1]/",
        "http://[fd00::1]/",
        "http://[::ffff:127.0.0.1]/",
        "http://[::ffff:10.0.0.1]/",
        "http://224.0.0.1/",
        "http://[ff02::1]/",
        "http://240.0.0.1/",
        "http://100.64.0.1/",
        "http://[fe80::1%25eth0]/",
        "http://backend/",
        "http://api.local/",
        "http://api.internal/",
        "http://api.svc/",
        "http://api.default.svc.cluster.local/",
        "http://API.INTERNAL./",
        "ftp://example.com/",
        "http:///missing-host",
        "not-a-url",
        "http://bad host.example/",
        "http://example.com:bad/",
    ],
)
async def test_rejected_destinations_never_reach_http(url):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await _check_single_url(client, url, 1.0)

    assert not result.valid
    assert result.error == "URL not allowed"
    assert result.status_code is None
    assert requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "addresses",
    [
        [],
        ["127.0.0.1"],
        ["10.0.0.1"],
        ["169.254.169.254"],
        ["::1"],
        ["93.184.216.34", "192.168.1.1"],
        ["93.184.216.34", "fd00::1"],
    ],
)
async def test_dns_answers_must_all_be_public(public_dns, addresses):
    public_dns.return_value = [
        (
            socket.AF_INET6 if ":" in address else socket.AF_INET,
            socket.SOCK_STREAM,
            6,
            "",
            (address, 443),
        )
        for address in addresses
    ]
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await _check_single_url(client, "https://docs.langchain.com/test", 1.0)

    assert not result.valid
    assert result.error == "URL not allowed"
    assert requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [socket.gaierror("private DNS details"), TimeoutError()]
)
async def test_dns_failures_are_generic(public_dns, error):
    public_dns.side_effect = error
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await _check_single_url(client, "https://unresolved.example.com/", 1.0)

    assert not result.valid
    assert result.error == "URL not allowed"
    assert requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,methods",
    [
        ("https://docs.langchain.com/overview", ["GET"]),
        ("https://example.com/overview", ["HEAD", "GET"]),
    ],
)
async def test_public_requests_are_pinned_including_fallback(public_dns, url, methods):
    requests = []

    def respond(request):
        requests.append(request)
        if request.method == "HEAD":
            return httpx.Response(405)
        return httpx.Response(200, text="<title>Documentation</title>")

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await _check_single_url(client, url, 1.0)

    assert result.valid
    assert result.final_url is None
    assert [request.method for request in requests] == methods
    assert all(request.url.host == "93.184.216.34" for request in requests)
    assert all(request.headers["host"] == httpx.URL(url).host for request in requests)
    assert all(
        request.extensions["sni_hostname"] == httpx.URL(url).host
        for request in requests
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,head_status",
    [
        ("https://docs.langchain.com/redirect", 302),
        ("https://example.com/redirect", 302),
        ("https://example.com/redirect", 405),
    ],
)
async def test_redirects_to_localhost_are_not_followed(url, head_status):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            head_status if request.method == "HEAD" else 302,
            headers={"Location": "http://localhost:8000/metrics"},
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond),
        follow_redirects=True,
    ) as client:
        result = await _check_single_url(client, url, 1.0)

    assert not result.valid
    assert result.status_code == 302
    assert result.final_url is None
    assert all(request.url.host == "93.184.216.34" for request in requests)
    assert len(requests) == (2 if head_status == 405 else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("address", ["93.184.216.34", "2606:4700:4700::1111"])
async def test_connection_uses_validated_ip_and_original_tls_hostname(
    monkeypatch, public_dns, address
):
    public_dns.return_value = [
        (
            socket.AF_INET6 if ":" in address else socket.AF_INET,
            socket.SOCK_STREAM,
            6,
            "",
            (address, 8443),
        ),
    ]
    connections = []
    tls_hosts = []
    sent_data = []

    class FakeNetworkStream:
        async def read(self, max_bytes, timeout=None):
            return b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n"

        async def write(self, buffer, timeout=None):
            sent_data.append(buffer)

        async def aclose(self):
            pass

        async def start_tls(self, ssl_context, server_hostname=None, timeout=None):
            assert ssl_context.check_hostname
            assert ssl_context.verify_mode == ssl.CERT_REQUIRED
            tls_hosts.append(server_hostname)
            return self

        def get_extra_info(self, info):
            return None

    async def connect_tcp(self, host, port, **kwargs):
        connections.append((host, port))
        public_dns.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port)),
        ]
        return FakeNetworkStream()

    monkeypatch.setattr(httpcore._backends.auto.AutoBackend, "connect_tcp", connect_tcp)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:8000")
    results = await _check_urls_async(["https://example.com:8443/path"], 1.0)

    assert results[0].valid
    assert connections == [(address, 8443)]
    assert tls_hosts == ["example.com"]
    assert b"Host: example.com:8443\r\n" in b"".join(sent_data)
