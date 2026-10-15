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
from unittest.mock import AsyncMock, MagicMock, patch

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
        self._content = content
        self.headers = {}

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
        LinkCheckResult(url="not-a-url", valid=False, error="Host not allowed"),
    ]

    with patch(
        "src.tools.link_check_tools._check_urls_async",
        new=_make_async_check_mock(fake_results),
    ):
        result = check_links.invoke({"urls": ["not-a-url"]})

    assert "0/1 valid" in result
    assert "Host not allowed" in result


@pytest.mark.asyncio
async def test_check_links_async_invalid_url_format():
    """Invalid URL in async context should also be reported correctly."""
    fake_results = [
        LinkCheckResult(
            url="ftp://not-supported", valid=False, error="Host not allowed"
        ),
    ]

    with patch(
        "src.tools.link_check_tools._check_urls_async",
        new=_make_async_check_mock(fake_results),
    ):
        result = await check_links.acall({"urls": ["ftp://not-supported"]})

    assert "0/1 valid" in result
    assert "Host not allowed" in result


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


@pytest.fixture
def network(monkeypatch):
    resolver = AsyncMock(
        return_value=[
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ]
    )
    monkeypatch.setattr(asyncio.BaseEventLoop, "getaddrinfo", resolver)
    requests = []
    responses = []

    async def handle_request(transport, request):
        requests.append(request)
        if responses:
            return responses.pop(0)
        return httpx.Response(200, text="<title>Documentation</title>")

    monkeypatch.setattr(
        httpx.AsyncHTTPTransport, "handle_async_request", handle_request
    )
    return resolver, requests, responses


@pytest.mark.asyncio
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize(
    "url",
    [
        "http://loki:8000/ok",
        "http://127.0.0.1",
        "http://169.254.169.254/latest/meta-data",
        "http://[::1]/",
        "http://2130706433/",
        "https://langchain.com.attacker.example/",
        "https://notlangchain.com/",
        "https://docs.langchain.com@loki/",
        "ftp://docs.langchain.com/",
    ],
)
async def test_disallowed_hosts_never_reach_dns_or_http(url, cached, network):
    resolver, requests, _ = network
    if cached:
        _cache[url] = LinkCheckResult(url=url, valid=True)

    result = (await _check_urls_async([url], 1.0))[0]

    assert result == LinkCheckResult(url=url, valid=False, error="Host not allowed")
    resolver.assert_not_awaited()
    assert requests == []
    if not cached:
        assert url not in _cache


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.1",
        "169.254.169.254",
        "0.0.0.0",
        "224.0.0.1",
        "240.0.0.1",
        "100.64.0.1",
        "::1",
        "fe80::1",
        "fc00::1",
        "ff02::1",
        "::",
        "::ffff:127.0.0.1",
    ],
)
async def test_any_nonpublic_dns_answer_blocks_request(address, network):
    resolver, requests, _ = network
    resolver.return_value.append(
        (socket.AF_INET6, socket.SOCK_STREAM, 6, "", (address, 443))
    )
    url = "https://docs.langchain.com/"

    result = (await _check_urls_async([url], 1.0))[0]

    assert result == LinkCheckResult(url=url, valid=False, error="Host not allowed")
    assert requests == []
    assert url not in _cache


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", [None, socket.gaierror("private DNS detail"), TimeoutError()]
)
async def test_dns_failure_is_closed_and_not_cached(failure, network):
    resolver, requests, _ = network
    resolver.return_value = []
    resolver.side_effect = failure
    url = "https://docs.langchain.com/"

    result = (await _check_urls_async([url], 1.0))[0]

    assert result == LinkCheckResult(url=url, valid=False, error="Host not allowed")
    assert "private DNS detail" not in _format_results([result])
    assert requests == []
    assert url not in _cache


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,fallback",
    [
        ("https://docs.langchain.com/start", False),
        ("https://github.com/langchain-ai", False),
        ("https://github.com/langchain-ai", True),
    ],
)
@pytest.mark.parametrize(
    "target",
    [
        "http://loki:8000/ok",
        "http://169.254.169.254/",
        "https://private.langchain.com/",
    ],
)
async def test_redirect_destinations_are_validated(url, fallback, target, network):
    resolver, requests, responses = network
    if target.startswith("https://private"):
        resolver.side_effect = [resolver.return_value] * (2 if fallback else 1) + [
            [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", 443))]
        ]
    if fallback:
        responses.append(httpx.Response(405))
    responses.append(httpx.Response(302, headers={"Location": target}))

    result = (await _check_urls_async([url], 1.0))[0]

    assert result == LinkCheckResult(url=url, valid=False, error="Host not allowed")
    assert len(requests) == (2 if fallback else 1)
    assert url not in _cache


@pytest.mark.asyncio
@pytest.mark.parametrize("address", ["93.184.216.34", "2606:4700::1111"])
async def test_allowed_docs_pin_connection_and_preserve_host_and_sni(address, network):
    resolver, requests, responses = network
    resolver.return_value = [
        (socket.AF_INET6, socket.SOCK_STREAM, 6, "", (address, 443))
    ]
    responses.extend(
        [
            httpx.Response(302, headers={"Location": "/final"}),
            httpx.Response(200, text="<title>Documentation</title>"),
        ]
    )
    url = "https://docs.langchain.com/start"

    result = (await _check_urls_async([url], 1.0))[0]

    assert result.valid
    assert result.final_url == "https://docs.langchain.com/final"
    assert [request.url.path for request in requests] == ["/start", "/final"]
    for request in requests:
        assert request.url.host == address
        assert request.headers["host"] == "docs.langchain.com"
        assert request.extensions["sni_hostname"] == "docs.langchain.com"


@pytest.mark.asyncio
async def test_redirect_limit(network):
    _, requests, responses = network
    responses.extend(
        httpx.Response(302, headers={"Location": "/next"}) for _ in range(6)
    )

    result = (await _check_urls_async(["https://docs.langchain.com/start"], 1.0))[0]

    assert not result.valid
    assert result.error == "Too many redirects"
    assert len(requests) == 6


@pytest.mark.parametrize(
    "host",
    [
        "langchain.com",
        "docs.langchain.com",
        "github.com",
        "langchain-ai.github.io",
        "pypi.org",
        "npmjs.com",
    ],
)
def test_allowed_host_suffixes(host):
    assert _is_valid_url(f"https://{host}/")


@pytest.mark.asyncio
async def test_http_connection_uses_pinned_ip_with_original_tls_host(monkeypatch):
    resolver = AsyncMock(
        return_value=[
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ]
    )
    monkeypatch.setattr(asyncio.BaseEventLoop, "getaddrinfo", resolver)
    stream = MagicMock()
    stream.get_extra_info.return_value = None
    stream.read = AsyncMock(
        return_value=b"HTTP/1.1 200 OK\r\nContent-Length: 18\r\n\r\n<title>Doc</title>"
    )
    stream.write = AsyncMock()
    stream.aclose = AsyncMock()
    stream.start_tls = AsyncMock(return_value=stream)
    with patch(
        "httpcore._backends.auto.AutoBackend.connect_tcp",
        new=AsyncMock(return_value=stream),
    ) as connect:
        result = (await _check_urls_async(["https://docs.langchain.com/"], 1.0))[0]

    assert result.valid
    assert connect.call_args.kwargs["host"] == "93.184.216.34"
    assert stream.start_tls.call_args.kwargs["server_hostname"] == "docs.langchain.com"
    context = stream.start_tls.call_args.kwargs["ssl_context"]
    assert context.check_hostname
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert b"Host: docs.langchain.com\r\n" in b"".join(
        call.args[0] for call in stream.write.call_args_list
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fallback,status", [(False, 200), (False, 404), (True, 200), (True, 500)]
)
async def test_permitted_head_and_get_fallback_preserve_status(
    fallback, status, network
):
    _, requests, responses = network
    if fallback:
        responses.append(httpx.Response(405))
    responses.append(httpx.Response(status))

    result = (await _check_urls_async(["https://github.com/langchain-ai"], 1.0))[0]

    assert result.valid == (status == 200)
    assert result.status_code == status
    assert [request.method for request in requests] == (
        ["HEAD", "GET"] if fallback else ["HEAD"]
    )


@pytest.mark.asyncio
async def test_rejected_destination_is_rechecked_not_cached(network):
    resolver, requests, _ = network
    public_addresses = resolver.return_value
    resolver.return_value = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", 443))
    ]
    url = "https://docs.langchain.com/"
    rejected = (await _check_urls_async([url], 1.0))[0]
    resolver.return_value = public_addresses

    permitted = (await _check_urls_async([url], 1.0))[0]

    assert rejected.error == "Host not allowed"
    assert permitted.valid
    assert len(requests) == 1
