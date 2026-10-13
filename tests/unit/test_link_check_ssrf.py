import asyncio
import socket
import ssl
from unittest.mock import AsyncMock

import httpx
import pytest

from src.tools import link_check_tools

PUBLIC_ADDRESS = "93.184.216.34"
DOCS_URL = "https://docs.langchain.com/oss/python/langchain/overview"


def _dns_answer(address=PUBLIC_ADDRESS):
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    return [(family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (address, 443))]


@pytest.fixture(autouse=True)
def public_dns(monkeypatch):
    link_check_tools._cache.clear()
    resolver = AsyncMock(return_value=_dns_answer())
    monkeypatch.setattr(asyncio.BaseEventLoop, "getaddrinfo", resolver)
    yield resolver
    link_check_tools._cache.clear()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8080/",
        "http://169.254.169.254/latest/meta-data/",
        "https://93.184.216.34/",
        "http://[::1]/",
        "http://[2606:4700:4700::1111]/",
        "https://callback.example/",
        "https://docs.langchain.com:8080/",
        "https://docs.langchain.com:80/",
        "https://user:password@docs.langchain.com/",
        "https://github.com/other-org/project",
        "https://github.com/langchain-ai/../other-org/project",
        "https://github.com/langchain-ai/%2e%2e/other-org/project",
        "https://github.com/langchain-ai/%252e%252e/other-org/project",
        "https://docs.langchain.com.evil.example/",
    ],
)
async def test_disallowed_urls_never_request_or_cache(url, public_dns):
    request = AsyncMock()
    link_check_tools._cache[url] = link_check_tools.LinkCheckResult(url=url, valid=True)
    async with httpx.AsyncClient(transport=httpx.MockTransport(request)) as client:
        result = await link_check_tools._check_single_url(client, url, 1.0)
    assert not result.valid
    assert result.error == "Host not allowed"
    request.assert_not_called()
    public_dns.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "not-a-url",
        "ftp://docs.langchain.com/",
        "https://",
        "https://docs.langchain.com:bad/",
        "https://docs.langchain.com/\n",
    ],
)
async def test_invalid_formats_never_request_or_cache(url, public_dns):
    request = AsyncMock()
    async with httpx.AsyncClient(transport=httpx.MockTransport(request)) as client:
        result = await link_check_tools._check_single_url(client, url, 1.0)
    assert result.error == "Invalid URL format"
    assert not link_check_tools._cache
    request.assert_not_called()
    public_dns.assert_not_called()


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
        "::",
        "fc00::1",
        "fe80::1",
        "ff02::1",
        "::ffff:127.0.0.1",
    ],
)
async def test_nonpublic_dns_answers_never_request(address, public_dns):
    public_dns.return_value = _dns_answer() + _dns_answer(address)
    request = AsyncMock()
    async with httpx.AsyncClient(transport=httpx.MockTransport(request)) as client:
        result = await link_check_tools._check_single_url(client, DOCS_URL, 1.0)
    assert result.error == "Host not allowed"
    assert not link_check_tools._cache
    request.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [[], OSError("DNS failed"), TimeoutError()])
async def test_failed_dns_never_requests(failure, public_dns):
    if isinstance(failure, Exception):
        public_dns.side_effect = failure
    else:
        public_dns.return_value = failure
    request = AsyncMock()
    async with httpx.AsyncClient(transport=httpx.MockTransport(request)) as client:
        result = await link_check_tools._check_single_url(client, DOCS_URL, 1.0)
    assert result.error == "Host not allowed"
    assert not link_check_tools._cache
    request.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "destination",
    [
        "https://callback.example/",
        "http://169.254.169.254/",
        "https://github.com/other-org/project",
        "https://docs.langchain.com:8080/",
    ],
)
async def test_redirect_rejects_destination_without_request(destination):
    requests = []

    def respond(request):
        requests.append(str(request.url))
        return httpx.Response(302, headers={"Location": destination})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond), follow_redirects=True
    ) as client:
        result = await link_check_tools._check_single_url(client, DOCS_URL, 1.0)
    assert result.error == "Host not allowed"
    assert requests == [DOCS_URL]
    assert not link_check_tools._cache


@pytest.mark.asyncio
async def test_redirect_rejects_private_dns_without_request(public_dns):
    public_dns.side_effect = [_dns_answer(), _dns_answer("10.0.0.1")]
    request = AsyncMock(
        return_value=httpx.Response(
            302, headers={"Location": "https://support.langchain.com/article"}
        )
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(request)) as client:
        result = await link_check_tools._check_single_url(client, DOCS_URL, 1.0)
    assert result.error == "Host not allowed"
    assert request.call_count == 1
    assert not link_check_tools._cache


@pytest.mark.asyncio
async def test_normalized_cache_is_revalidated(public_dns):
    request = AsyncMock(
        return_value=httpx.Response(200, text="<title>LangChain</title>")
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(request)) as client:
        first = await link_check_tools._check_single_url(
            client, "https://DOCS.langchain.com:443/guide#section", 1.0
        )
        second = await link_check_tools._check_single_url(
            client, "https://docs.langchain.com/guide", 1.0
        )
        public_dns.return_value = _dns_answer("127.0.0.1")
        third = await link_check_tools._check_single_url(
            client, "https://docs.langchain.com/guide", 1.0
        )
    assert first.valid and second.valid
    assert third.error == "Host not allowed"
    assert request.call_count == 1
    assert public_dns.call_count == 3
    assert list(link_check_tools._cache) == ["https://docs.langchain.com/guide"]
    assert second.url == "https://docs.langchain.com/guide"


@pytest.mark.asyncio
async def test_relative_redirects_and_soft_404_are_checked():
    requests = []

    def respond(request):
        requests.append(str(request.url))
        if len(requests) == 1:
            return httpx.Response(302, headers={"Location": "/missing"})
        return httpx.Response(200, text="<title>Page not found</title>")

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await link_check_tools._check_single_url(client, DOCS_URL, 1.0)
    assert result.error == "Soft 404: Page shows 'not found' content"
    assert result.final_url == "https://docs.langchain.com/missing"
    assert requests == [DOCS_URL, result.final_url]
    assert not link_check_tools._cache


@pytest.mark.asyncio
async def test_redirect_limit():
    request = AsyncMock(return_value=httpx.Response(302, headers={"Location": "/loop"}))
    async with httpx.AsyncClient(transport=httpx.MockTransport(request)) as client:
        result = await link_check_tools._check_single_url(client, DOCS_URL, 1.0)
    assert result.error == "Too many redirects"
    assert request.call_count == link_check_tools.MAX_REDIRECTS + 1
    assert not link_check_tools._cache


@pytest.mark.asyncio
async def test_head_get_fallback_and_redirect_to_docs():
    methods = []

    def respond(request):
        methods.append((request.method, str(request.url)))
        if request.method == "HEAD":
            return httpx.Response(405)
        if request.url.host == "github.com":
            return httpx.Response(302, headers={"Location": DOCS_URL})
        return httpx.Response(200, text="<title>LangChain docs</title>")

    github_url = "https://github.com/langchain-ai/langchain"
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond), follow_redirects=True
    ) as client:
        result = await link_check_tools._check_single_url(client, github_url, 1.0)
    assert result.valid
    assert result.final_url == DOCS_URL
    assert methods == [("HEAD", github_url), ("GET", github_url), ("GET", DOCS_URL)]


@pytest.mark.asyncio
async def test_timeout_preserves_error():
    request = AsyncMock(side_effect=httpx.ReadTimeout("timeout"))
    async with httpx.AsyncClient(transport=httpx.MockTransport(request)) as client:
        result = await link_check_tools._check_single_url(client, DOCS_URL, 1.0)
    assert result.error == "Request timed out"


@pytest.mark.asyncio
async def test_production_transport_pins_ip_and_tls_identity(monkeypatch, public_dns):
    requests = []

    async def respond(transport, request):
        requests.append(request)
        assert transport._pool._max_keepalive_connections == 0
        assert isinstance(transport._pool._ssl_context, ssl.SSLContext)
        assert transport._pool._ssl_context.check_hostname
        assert transport._pool._ssl_context.verify_mode == ssl.CERT_REQUIRED
        return httpx.Response(200, text="<title>LangChain docs</title>")

    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:8080")
    monkeypatch.setenv("ALL_PROXY", "http://169.254.169.254:8080")
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", respond)
    results = await link_check_tools._check_urls_async([DOCS_URL], 1.0)
    assert results[0].valid
    public_dns.assert_awaited_once()
    assert len(requests) == 1
    assert requests[0].url.host == PUBLIC_ADDRESS
    assert requests[0].headers["Host"] == "docs.langchain.com"
    assert requests[0].extensions["sni_hostname"] == "docs.langchain.com"
    assert requests[0].url.path == httpx.URL(DOCS_URL).path


@pytest.mark.asyncio
@pytest.mark.parametrize("address", [PUBLIC_ADDRESS, "2606:4700:4700::1111"])
async def test_actual_connection_uses_validated_ip_without_dns_rebinding(
    address, monkeypatch, public_dns
):
    public_dns.side_effect = [_dns_answer(address), _dns_answer("127.0.0.1")]
    connections = []
    tls_hostnames = []
    writes = []

    class FakeNetworkStream:
        async def read(self, max_bytes, timeout=None):
            return b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK"

        async def write(self, buffer, timeout=None):
            writes.append(buffer)

        async def start_tls(self, ssl_context, server_hostname=None, timeout=None):
            assert ssl_context.check_hostname
            assert ssl_context.verify_mode == ssl.CERT_REQUIRED
            tls_hostnames.append(server_hostname)
            return self

        async def aclose(self):
            pass

        def get_extra_info(self, info):
            return None

    async def connect(backend, host, port, **kwargs):
        connections.append((host, port))
        return FakeNetworkStream()

    monkeypatch.setattr("httpcore._backends.anyio.AnyIOBackend.connect_tcp", connect)
    results = await link_check_tools._check_urls_async([DOCS_URL], 1.0)

    assert results[0].valid
    public_dns.assert_awaited_once()
    assert connections == [(address, 443)]
    assert tls_hostnames == ["docs.langchain.com"]
    assert b"Host: docs.langchain.com\r\n" in b"".join(writes)
