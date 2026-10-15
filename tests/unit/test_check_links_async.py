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

import httpx
import pytest

# ---------------------------------------------------------------------------
# Helpers to build fake LinkCheckResult objects without importing the whole
# module (which would trigger import-time side effects).
# ---------------------------------------------------------------------------
from src.tools.link_check_tools import (
    CONTENT_CHECK_BYTES,
    LinkCheckResult,
    _cache,
    _check_single_url,
    _check_urls_async,
    _format_results,
    _is_valid_url,
    check_links,
)


class _FakeStreamResponse:
    """Minimal async streaming response for _check_single_url tests."""

    def __init__(self, url: str, status_code: int, content: str):
        self.url = url
        self.status_code = status_code
        self.is_redirect = False
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
    """The tool must be awaitable inside an existing event loop."""
    fake_results = [
        LinkCheckResult(url="https://example.com", valid=True, status_code=200),
    ]

    with patch(
        "src.tools.link_check_tools._check_urls_async",
        new=_make_async_check_mock(fake_results),
    ):
        result = await check_links.ainvoke({"urls": ["https://example.com"]})

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
            await check_links.ainvoke({"urls": ["https://docs.langchain.com/"]})
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
    """A synchronous caller can run the async tool in its own event loop."""
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
        result = asyncio.run(
            check_links.ainvoke(
                {"urls": ["https://example.com", "https://bad.example.com"]}
            )
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
        result = asyncio.run(
            check_links.ainvoke(
                {
                    "urls": [
                        "https://example.com",
                        "https://example.com",
                        "https://example.com",
                    ]
                }
            )
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
    result = asyncio.run(check_links.ainvoke({"urls": []}))
    assert result == "No URLs provided to check."


@pytest.mark.asyncio
async def test_check_links_async_empty_list():
    """Empty list in async context should also return the 'no URLs' message."""
    result = await check_links.ainvoke({"urls": []})
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
        result = asyncio.run(check_links.ainvoke({"urls": ["not-a-url"]}))

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
        result = await check_links.ainvoke({"urls": ["ftp://not-supported"]})

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


@pytest.fixture(autouse=True)
def clear_link_cache():
    _cache.clear()
    yield
    _cache.clear()


def _dns_answer(address="93.184.216.34"):
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    return (family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (address, 443))


@pytest.fixture
def network(monkeypatch):
    dns = AsyncMock(return_value=[_dns_answer()])
    send = AsyncMock(return_value=httpx.Response(200, text="Useful documentation"))
    monkeypatch.setattr(asyncio.BaseEventLoop, "getaddrinfo", dns)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", send)
    monkeypatch.setenv("HTTPS_PROXY", "http://vault:8000")
    return dns, send


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://vault:8000/ok",
        "http://169.254.169.254/latest/meta-data/",
        "http://127.0.0.1/",
        "http://[::1]/",
        "https://notlangchain.com/",
        "https://docs.langchain.com.evil.org/",
        "https://docs.langchain.com:8000/",
        "http://docs.langchain.com:443/",
        "https://docs.langchain.com:invalid/",
        "https://docs.langchain.com:/",
        "https://docs.langchain.com@vault/",
        "https://vault@docs.langchain.com/",
        "https://docs.langchain.com\\@vault/",
        "https://docs.langchain.com\n/",
        "ftp://docs.langchain.com/",
        "not-a-url",
    ],
)
async def test_blocked_destinations_make_no_network_calls(network, url):
    dns, send = network
    (result,) = await _check_urls_async([url], 1.0)
    assert not result.valid
    assert result.error == "Host not allowed"
    dns.assert_not_called()
    send.assert_not_called()


@pytest.mark.parametrize(
    "url",
    [
        "https://docs.langchain.com/",
        "https://DOCS.LANGCHAIN.COM:443/",
        "http://docs.langchain.com:80/",
        "https://api.python.langchain.com/",
        "https://langchain-ai.github.io/",
        "https://github.com/langchain-ai/langchain",
    ],
)
def test_allowed_hostnames(url):
    assert _is_valid_url(url)


@pytest.mark.asyncio
@pytest.mark.parametrize("address", ["93.184.216.34", "2606:4700:4700::1111"])
async def test_public_destination_is_pinned_with_original_hostname(network, address):
    dns, send = network
    dns.return_value = [_dns_answer(address)]
    url = "https://docs.langchain.com/guide"
    (result,) = await _check_urls_async([url], 1.0)
    assert result.valid
    assert result.final_url is None
    assert dns.call_args.args[0] == "docs.langchain.com"
    request = send.call_args.args[0]
    assert request.url.host == address
    assert request.headers["host"] == "docs.langchain.com"
    assert request.extensions["sni_hostname"] == "docs.langchain.com"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "address",
    [
        "10.0.0.1",
        "172.16.0.1",
        "192.168.0.1",
        "127.0.0.1",
        "169.254.169.254",
        "224.0.0.1",
        "240.0.0.1",
        "0.0.0.0",
        "100.64.0.1",
        "::1",
        "::",
        "fc00::1",
        "fe80::1",
        "ff02::1",
        "::ffff:127.0.0.1",
    ],
)
async def test_any_prohibited_dns_answer_blocks_connection(network, address):
    dns, send = network
    dns.return_value = [_dns_answer(), _dns_answer(address)]
    (result,) = await _check_urls_async(["https://docs.langchain.com/"], 1.0)
    assert not result.valid
    assert result.error == "Host not allowed"
    send.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source,fallback",
    [
        ("https://docs.langchain.com/", False),
        ("https://github.com/", False),
        ("https://github.com/", True),
    ],
)
async def test_blocked_redirect_is_not_resolved_or_requested(network, source, fallback):
    dns, send = network
    responses = [httpx.Response(302, headers={"location": "http://vault:8000/ok"})]
    if fallback:
        responses.insert(0, httpx.Response(405))
    send.side_effect = responses
    (result,) = await _check_urls_async([source], 1.0)
    assert result.error == "Host not allowed"
    assert not result.valid
    assert all(call.args[0] == httpx.URL(source).host for call in dns.call_args_list)
    assert send.call_count == len(responses)


@pytest.mark.asyncio
async def test_allowed_redirects_and_head_fallback(network):
    dns, send = network
    send.side_effect = [
        httpx.Response(405),
        httpx.Response(302, headers={"location": "https://docs.langchain.com/old"}),
        httpx.Response(301, headers={"location": "/new"}),
        httpx.Response(200, text="Documentation"),
    ]
    (result,) = await _check_urls_async(["https://github.com/"], 1.0)
    assert result.valid
    assert result.final_url == "https://docs.langchain.com/new"
    assert [call.args[0].method for call in send.call_args_list] == [
        "HEAD",
        "GET",
        "GET",
        "GET",
    ]
    assert dns.call_count == 4


@pytest.mark.asyncio
async def test_dns_rebinding_is_blocked_on_fallback(network):
    dns, send = network
    dns.side_effect = [[_dns_answer()], [_dns_answer("127.0.0.1")]]
    send.return_value = httpx.Response(405)
    (result,) = await _check_urls_async(["https://github.com/"], 1.0)
    assert not result.valid
    assert result.error == "Host not allowed"
    assert send.call_count == 1


@pytest.mark.asyncio
async def test_redirect_to_allowed_hostname_with_private_dns_is_blocked(network):
    dns, send = network
    dns.side_effect = [[_dns_answer()], [_dns_answer("10.0.0.1")]]
    send.return_value = httpx.Response(
        302, headers={"location": "https://support.langchain.com/"}
    )
    (result,) = await _check_urls_async(["https://docs.langchain.com/"], 1.0)
    assert not result.valid
    assert result.error == "Host not allowed"
    assert send.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        OSError("private resolver details"),
        httpx.ConnectError("connection to internal-auth:8000 failed"),
        httpx.ReadTimeout("sensitive timeout details"),
    ],
)
async def test_transient_errors_are_generic_and_not_cached(network, failure):
    dns, send = network
    if isinstance(failure, OSError):
        dns.side_effect = [failure, [_dns_answer()]]
    else:
        send.side_effect = [failure, httpx.Response(200)]
    url = "https://docs.langchain.com/"
    (first,) = await _check_urls_async([url], 1.0)
    assert not first.valid
    assert first.error == "unreachable"
    assert str(failure) not in _format_results([first])
    (second,) = await _check_urls_async([url], 1.0)
    assert second.valid


@pytest.mark.asyncio
async def test_redirect_limit(network):
    _, send = network
    send.side_effect = lambda request: httpx.Response(
        302, headers={"location": "/loop"}
    )
    (result,) = await _check_urls_async(["https://docs.langchain.com/"], 1.0)
    assert result.error == "Too many redirects"
    assert send.call_count == 6


@pytest.mark.asyncio
async def test_soft_404_stream_stops_at_content_limit(network):
    _, send = network

    class ContentStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"<title>404 not found</title>" + b" " * CONTENT_CHECK_BYTES
            pytest.fail("The link checker read beyond its content limit")

    send.return_value = httpx.Response(200, stream=ContentStream())
    (result,) = await _check_urls_async(["https://docs.langchain.com/"], 1.0)
    assert not result.valid
    assert result.error == "Soft 404: Page shows 'not found' content"


@pytest.mark.asyncio
async def test_socket_and_tls_use_pinned_ip_and_original_identity(monkeypatch):
    connections = []
    tls_hosts = []
    writes = []

    class NetworkStream:
        async def read(self, max_bytes, timeout=None):
            return b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n"

        async def write(self, buffer, timeout=None):
            writes.append(buffer)

        async def aclose(self):
            pass

        async def start_tls(self, ssl_context, server_hostname=None, timeout=None):
            assert ssl_context.check_hostname
            assert ssl_context.verify_mode == ssl.CERT_REQUIRED
            tls_hosts.append(server_hostname)
            return self

        def get_extra_info(self, info):
            return None

    async def connect_tcp(host, port, **kwargs):
        connections.append((host, port))
        await asyncio.sleep(0)
        return NetworkStream()

    monkeypatch.setattr(
        asyncio.BaseEventLoop, "getaddrinfo", AsyncMock(return_value=[_dns_answer()])
    )
    with patch(
        "httpcore._backends.auto.AutoBackend.connect_tcp", side_effect=connect_tcp
    ):
        results = await _check_urls_async(
            ["https://github.com/", "https://langchain.com/"], 1.0
        )
    assert all(result.valid for result in results)
    assert connections == [("93.184.216.34", 443)] * 2
    assert sorted(tls_hosts) == ["github.com", "langchain.com"]
    assert any(b"Host: github.com\r\n" in data for data in writes)
    assert any(b"Host: langchain.com\r\n" in data for data in writes)
