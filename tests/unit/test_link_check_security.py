"""Test destination policy, redirects, and pinned link-check connections."""

import asyncio
import socket
import ssl
from unittest.mock import AsyncMock

import httpcore
import httpx
import pytest

from src.tools import link_check_tools as links

PUBLIC_IP = "93.184.216.34"


def addresses(*values):
    return [
        (
            socket.AF_INET6 if ":" in value else socket.AF_INET,
            socket.SOCK_STREAM,
            6,
            "",
            (value, 443),
        )
        for value in values
    ]


@pytest.fixture(autouse=True)
def clear_cache():
    links._cache.clear()
    yield
    links._cache.clear()


@pytest.fixture
def dns(monkeypatch):
    resolver = AsyncMock(return_value=addresses(PUBLIC_IP))

    async def run(callback):
        monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", resolver)
        return await callback()

    return resolver, run


def check(url, dns, handler):
    resolver, run = dns
    requests = []

    def record(request):
        requests.append(request)
        return handler(request)

    async def call():
        async with httpx.AsyncClient(transport=httpx.MockTransport(record)) as client:
            return await links._check_single_url(client, url, 1.0)

    result = asyncio.run(run(call))
    return result, requests, resolver


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8080/",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.1.2.3/",
        "http://[::1]/",
        "https://example.com/",
        "https://evillangchain.com/",
        "https://langchain.com.evil.test/",
        "https://github.com/other/repo",
        "https://github.com/langchain-ai-evil/repo",
        "https://github.com/langchain-ai/%2e%2e/other",
        "https://user:password@docs.langchain.com/",
    ],
)
def test_disallowed_targets_are_rejected_before_dns_or_http(url, dns):
    links._cache[url] = links.LinkCheckResult(url=url, valid=True)
    result, requests, resolver = check(
        url, dns, lambda request: pytest.fail("Unexpected HTTP request")
    )
    assert result == links.LinkCheckResult(
        url=url, valid=False, error="Host not allowed for link checking"
    )
    assert requests == []
    resolver.assert_not_awaited()


@pytest.mark.parametrize(
    "values",
    [
        [],
        ["127.0.0.1"],
        ["169.254.169.254"],
        ["10.1.2.3"],
        ["224.0.0.1"],
        ["240.0.0.1"],
        ["::1"],
        ["fe80::1"],
        ["fec0::1"],
        ["::ffff:127.0.0.1"],
        [PUBLIC_IP, "10.1.2.3"],
    ],
)
def test_unsafe_dns_answers_are_rejected_without_http(values, dns):
    dns[0].return_value = addresses(*values)
    result, requests, _ = check(
        "https://docs.langchain.com/",
        dns,
        lambda request: pytest.fail("Unexpected HTTP request"),
    )
    assert not result.valid
    assert result.error == "Host not allowed for link checking"
    assert requests == []


def test_dns_failure_is_rejected_without_http(dns):
    dns[0].side_effect = socket.gaierror("No address")
    result, requests, _ = check(
        "https://docs.langchain.com/",
        dns,
        lambda request: pytest.fail("Unexpected HTTP request"),
    )
    assert not result.valid
    assert requests == []


@pytest.mark.parametrize(
    "url",
    [
        "https://docs.langchain.com/overview",
        "https://support.langchain.com/articles/123",
        "https://academy.langchain.com/",
        "https://www.langchain.com/",
        "https://smith.langchain.com/",
        "https://smith.langchain.dev/",
        "https://langchain.com/",
        "https://langchain.dev/",
        "https://github.com/langchain-ai/langchain",
    ],
)
def test_approved_links_preserve_host_and_tls_identity(url, dns):
    result, requests, _ = check(
        url,
        dns,
        lambda request: httpx.Response(200, text="<title>Documentation</title>"),
    )
    assert result.valid
    assert result.status_code == 200
    assert result.final_url is None
    assert len(requests) == 1
    assert requests[0].url.host == PUBLIC_IP
    assert requests[0].headers["host"] == httpx.URL(url).host
    assert requests[0].extensions["sni_hostname"] == httpx.URL(url).host


@pytest.mark.parametrize(
    "location",
    [
        "http://169.254.169.254/latest/meta-data/",
        "https://external.test/",
        "//127.0.0.1:8080/",
    ],
)
def test_disallowed_redirect_is_never_contacted_or_disclosed(location, dns):
    url = "https://docs.langchain.com/start"
    result, requests, resolver = check(
        url, dns, lambda request: httpx.Response(302, headers={"Location": location})
    )
    assert not result.valid
    assert result.error == "Host not allowed for link checking"
    assert result.final_url is None
    assert len(requests) == 1
    assert resolver.await_count == 1
    assert url not in links._cache


def test_relative_redirect_is_checked_and_soft_404_detected(dns):
    def respond(request):
        if request.url.path == "/start":
            return httpx.Response(302, headers={"Location": "/missing"})
        return httpx.Response(200, text="<title>Page Not Found</title>")

    result, requests, resolver = check("https://docs.langchain.com/start", dns, respond)
    assert not result.valid
    assert result.error.startswith("Soft 404:")
    assert result.final_url == "https://docs.langchain.com/missing"
    assert len(requests) == resolver.await_count == 2


def test_head_redirect_to_documentation_uses_content_check(dns):
    def respond(request):
        if request.headers["host"] == "www.langchain.com":
            return httpx.Response(
                302, headers={"Location": "https://docs.langchain.com/missing"}
            )
        return httpx.Response(200, text="<title>404</title>")

    result, requests, _ = check("https://www.langchain.com/start", dns, respond)
    assert not result.valid
    assert [request.method for request in requests] == ["HEAD", "GET"]


def test_redirect_limit_is_bounded(dns):
    result, requests, _ = check(
        "https://docs.langchain.com/start",
        dns,
        lambda request: httpx.Response(302, headers={"Location": "/again"}),
    )
    assert result.error == "Too many redirects"
    assert len(requests) == links.MAX_REDIRECTS + 1


def test_head_to_get_fallback_revalidates_and_pins_dns(dns):
    dns[0].side_effect = [addresses(PUBLIC_IP), addresses("8.8.8.8")]

    def respond(request):
        return httpx.Response(405 if request.method == "HEAD" else 200)

    result, requests, resolver = check(
        "https://github.com/langchain-ai/langchain", dns, respond
    )
    assert result.valid
    assert [request.method for request in requests] == ["HEAD", "GET"]
    assert [request.url.host for request in requests] == [PUBLIC_IP, "8.8.8.8"]
    assert resolver.await_count == 2


@pytest.mark.parametrize("status", [302, 405])
def test_dns_rebinding_on_redirect_or_fallback_is_blocked(status, dns):
    dns[0].side_effect = [addresses(PUBLIC_IP), addresses("169.254.169.254")]
    result, requests, _ = check(
        "https://www.langchain.com/start",
        dns,
        lambda request: httpx.Response(status, headers={"Location": "/next"}),
    )
    assert result.error == "Host not allowed for link checking"
    assert result.final_url is None
    assert len(requests) == 1


@pytest.mark.parametrize("cached_redirect", [False, True])
def test_cache_cannot_bypass_dns_validation(cached_redirect, dns):
    url = "https://docs.langchain.com/"
    links._cache[url] = links.LinkCheckResult(
        url=url,
        valid=True,
        final_url="https://support.langchain.com/article" if cached_redirect else None,
    )
    dns[0].side_effect = (
        [addresses(PUBLIC_IP), addresses("10.1.2.3")]
        if cached_redirect
        else [addresses("10.1.2.3")]
    )
    result, requests, _ = check(
        url, dns, lambda request: pytest.fail("Unexpected HTTP request")
    )
    assert not result.valid
    assert result.final_url is None
    assert requests == []


@pytest.mark.parametrize("use_tool", [False, True])
def test_real_transport_connects_only_to_pinned_ips_with_original_tls_hosts(
    monkeypatch, dns, use_tool
):
    connections = []
    tls_hosts = []
    writes = []

    class Stream:
        async def read(self, max_bytes, timeout=None):
            if b"/redirect" in self.request:
                return b"HTTP/1.1 302 Found\r\nLocation: http://169.254.169.254/latest/meta-data/\r\nContent-Length: 0\r\n\r\n"
            return b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n"

        async def write(self, buffer, timeout=None):
            writes.append(buffer)
            if buffer:
                self.request = buffer

        async def start_tls(self, ssl_context, server_hostname=None, timeout=None):
            assert ssl_context.verify_mode == ssl.CERT_REQUIRED
            assert ssl_context.check_hostname
            tls_hosts.append(server_hostname)
            return self

        async def aclose(self):
            pass

        def get_extra_info(self, info):
            return None

    async def connect(self, host, port, **kwargs):
        connections.append((host, port))
        return Stream()

    monkeypatch.setattr(httpcore.AnyIOBackend, "connect_tcp", connect)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:8080")

    async def call():
        urls = [
            "https://docs.langchain.com/",
            "https://support.langchain.com/",
            "https://docs.langchain.com/redirect",
            "http://169.254.169.254/",
        ]
        if use_tool:
            return await links.check_links.ainvoke({"urls": urls, "timeout": 1.0})
        return await links._check_urls_async(urls, 1.0)

    results = asyncio.run(dns[1](call))
    if use_tool:
        assert "2/4 valid" in results
        assert "Host not allowed for link checking" in results
        assert "latest/meta-data" not in results
    else:
        assert [result.valid for result in results] == [True, True, False, False]
        assert results[2].final_url is None
    assert connections == [(PUBLIC_IP, 443)] * 3
    assert set(tls_hosts) == {"docs.langchain.com", "support.langchain.com"}
    assert any(b"Host: docs.langchain.com" in buffer for buffer in writes)
    assert any(b"Host: support.langchain.com" in buffer for buffer in writes)
