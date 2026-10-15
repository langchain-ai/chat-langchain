"""Exercise public-only link checking without network access."""

import asyncio
import socket
import ssl
from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio

from src.tools.link_check_tools import (
    MAX_REDIRECTS,
    MAX_URLS_PER_CALL,
    _cache,
    _check_single_url,
    _check_urls_async,
    check_links,
)


def _dns_records(*addresses):
    return [
        (
            socket.AF_INET6 if ":" in address else socket.AF_INET,
            socket.SOCK_STREAM,
            socket.IPPROTO_TCP,
            "",
            (address, 443),
        )
        for address in addresses
    ]


@pytest.fixture(autouse=True)
def clear_cache():
    _cache.clear()
    yield
    _cache.clear()


@pytest_asyncio.fixture
async def resolver(monkeypatch):
    resolve = AsyncMock(return_value=_dns_records("93.184.216.34"))
    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", resolve)
    return resolve


@pytest_asyncio.fixture
async def http_client():
    requests = []
    responses = []

    def handle(request):
        requests.append(request)
        return responses.pop(0) if responses else httpx.Response(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        yield client, requests, responses


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,address",
    [
        ("http://localhost", "127.0.0.1"),
        ("http://localhost/ok", "127.0.0.1"),
        ("http://localhost:8000/metrics?format=text", "::1"),
        ("http://localhost./nested/path", "127.0.0.1"),
        ("http://127.0.0.1:8080/ok", "127.0.0.1"),
        ("http://0.0.0.0/metrics", "0.0.0.0"),
        ("http://[::1]/ok", "::1"),
        ("http://[::]/", "::"),
        ("http://[::ffff:127.0.0.1]/", "::ffff:127.0.0.1"),
        ("http://metadata.google.internal/computeMetadata/v1/", "169.254.169.254"),
        ("http://internal-api:3000/metrics", "10.0.0.2"),
        ("http://service.namespace.svc.cluster.local/", "172.16.0.5"),
        ("http://192.168.1.1/", "192.168.1.1"),
        ("http://100.64.0.1/", "100.64.0.1"),
        ("http://224.0.0.1/", "224.0.0.1"),
        ("http://240.0.0.1/", "240.0.0.1"),
        ("http://[fc00::1]/", "fc00::1"),
        ("http://[fe80::1]/", "fe80::1"),
        ("http://[ff02::1]/", "ff02::1"),
        ("http://[2001:db8::1]/", "2001:db8::1"),
        ("http://2130706433/", "127.0.0.1"),
        ("http://0x7f000001/", "127.0.0.1"),
    ],
)
async def test_non_public_destinations_are_not_requested(
    url, address, resolver, http_client
):
    resolver.return_value = _dns_records(address)
    client, requests, _ = http_client
    result = await _check_single_url(client, url, 1.0)
    assert not result.valid
    assert result.error == "Blocked destination: non-public address"
    assert requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com/file",
        "http:///missing-host",
        "http://[::1",
        "http://example.com:invalid/",
        "http://user:password@example.com/",
        "http://[fe80::1%25eth0]/",
        "http://example.com\\@localhost/",
        "http://example.com/\nmetrics",
        "http://0177.0.0.1/",
    ],
)
async def test_malformed_urls_are_not_requested(url, resolver, http_client):
    client, requests, _ = http_client
    result = await _check_single_url(client, url, 1.0)
    assert not result.valid
    assert result.error == "Invalid URL format"
    assert requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "addresses", [[], ["93.184.216.34", "127.0.0.1"], ["93.184.216.34", "::1"]]
)
async def test_all_dns_answers_must_be_public(addresses, resolver, http_client):
    resolver.return_value = _dns_records(*addresses)
    client, requests, _ = http_client
    result = await _check_single_url(client, "https://example.com/", 1.0)
    assert not result.valid
    assert requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [socket.gaierror("unknown host"), TimeoutError()])
async def test_dns_failure_is_closed(error, resolver, http_client):
    resolver.side_effect = error
    client, requests, _ = http_client
    result = await _check_single_url(client, "https://example.com/", 1.0)
    assert not result.valid
    assert result.error == "Blocked destination: DNS resolution failed"
    assert requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "https://docs.langchain.com/start",
        "https://api.smith.langchain.com/info",
        "https://bücher.example/start",
    ],
)
@pytest.mark.parametrize("address", ["93.184.216.34", "2606:4700:4700::1111"])
async def test_public_links_preserve_host_and_tls_identity(
    url, address, resolver, http_client
):
    resolver.return_value = _dns_records(address)
    client, requests, _ = http_client
    result = await _check_single_url(client, url, 1.0)
    assert result.valid
    assert result.status_code == 200
    assert result.final_url is None
    assert len(requests) == 1
    assert requests[0].url.host == address
    assert requests[0].headers["host"] == httpx.URL(url).netloc.decode("ascii")
    assert requests[0].extensions["sni_hostname"] == httpx.URL(url).raw_host.decode(
        "ascii"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,fallback",
    [
        ("https://docs.langchain.com/start", False),
        ("https://example.com/start", False),
        ("https://example.com/start", True),
    ],
)
@pytest.mark.parametrize(
    "destination",
    [
        "http://127.0.0.1/metrics",
        "//127.0.0.1/metrics",
        "http://[::1]/ok",
        "http://metadata.google.internal/",
        "http://internal-api/metrics",
        "file:///etc/passwd",
    ],
)
async def test_redirects_cannot_reach_prohibited_destinations(
    url, fallback, destination, resolver, http_client
):
    resolver.side_effect = [_dns_records("93.184.216.34")] * (2 if fallback else 1) + [
        _dns_records("10.0.0.1")
    ]
    client, requests, responses = http_client
    if fallback:
        responses.append(httpx.Response(405))
    responses.append(httpx.Response(302, headers={"location": destination}))
    result = await _check_single_url(client, url, 1.0)
    assert not result.valid
    assert len(requests) == (2 if fallback else 1)
    assert all(request.url.host == "93.184.216.34" for request in requests)
    assert [request.method for request in requests] == (
        ["HEAD", "GET"]
        if fallback
        else ["GET" if "docs.langchain.com" in url else "HEAD"]
    )


@pytest.mark.asyncio
async def test_relative_redirects_preserve_public_final_url(resolver, http_client):
    client, requests, responses = http_client
    responses.extend(
        [httpx.Response(302, headers={"location": "../guide"}), httpx.Response(200)]
    )
    result = await _check_single_url(client, "https://example.com:8443/docs/start", 1.0)
    assert result.valid
    assert result.final_url == "https://example.com:8443/guide"
    assert [request.url.path for request in requests] == ["/docs/start", "/guide"]
    assert all(request.headers["host"] == "example.com:8443" for request in requests)


@pytest.mark.asyncio
@pytest.mark.parametrize("redirects", [MAX_REDIRECTS, MAX_REDIRECTS + 1])
async def test_redirect_limit(redirects, resolver, http_client):
    client, requests, responses = http_client
    responses.extend(
        httpx.Response(302, headers={"location": f"/hop/{index}"})
        for index in range(redirects)
    )
    result = await _check_single_url(client, "https://example.com/start", 1.0)
    assert result.valid == (redirects == MAX_REDIRECTS)
    assert len(requests) == MAX_REDIRECTS + 1
    if not result.valid:
        assert result.error == "Too many redirects"


@pytest.mark.asyncio
async def test_dns_rebinding_on_redirect_is_blocked(resolver, http_client):
    resolver.side_effect = [_dns_records("93.184.216.34"), _dns_records("127.0.0.1")]
    client, requests, responses = http_client
    responses.append(httpx.Response(302, headers={"location": "/metrics"}))
    result = await _check_single_url(client, "https://example.com/start", 1.0)
    assert not result.valid
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_dns_rebinding_on_get_fallback_is_blocked(resolver, http_client):
    resolver.side_effect = [
        _dns_records("93.184.216.34"),
        _dns_records("127.0.0.1"),
    ]
    client, requests, responses = http_client
    responses.append(httpx.Response(405))
    result = await _check_single_url(client, "https://example.com/start", 1.0)
    assert not result.valid
    assert [request.method for request in requests] == ["HEAD"]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [200, 404])
async def test_public_get_fallback_preserves_status(status, resolver, http_client):
    client, requests, responses = http_client
    responses.extend([httpx.Response(405), httpx.Response(status)])
    result = await _check_single_url(client, "http://example.com/start", 1.0)
    assert result.valid == (status == 200)
    assert result.status_code == status
    assert result.error == (None if status == 200 else "HTTP 404")
    assert [request.method for request in requests] == ["HEAD", "GET"]


@pytest.mark.asyncio
async def test_redirect_to_documentation_still_detects_soft_404(resolver, http_client):
    client, requests, responses = http_client
    responses.extend(
        [
            httpx.Response(
                302, headers={"location": "https://docs.langchain.com/missing"}
            ),
            httpx.Response(200, text="<title>Page not found</title>"),
        ]
    )
    result = await _check_single_url(client, "https://example.com/start", 1.0)
    assert not result.valid
    assert result.status_code == 200
    assert result.error == "Soft 404: Page shows 'not found' content"
    assert result.final_url == "https://docs.langchain.com/missing"
    assert [request.method for request in requests] == ["HEAD", "GET"]


@pytest.mark.asyncio
async def test_tool_checks_only_first_fifteen_unique_urls(
    resolver, http_client, monkeypatch
):
    client, requests, _ = http_client

    async def check_urls(urls, timeout):
        return list(
            await asyncio.gather(
                *(_check_single_url(client, url, timeout) for url in urls)
            )
        )

    monkeypatch.setattr("src.tools.link_check_tools._check_urls_async", check_urls)
    urls = [f"https://example.com/{index}" for index in range(MAX_URLS_PER_CALL + 3)]
    output = await check_links.ainvoke({"urls": [urls[0], *urls]})
    assert [request.url.path for request in requests] == [
        f"/{index}" for index in range(MAX_URLS_PER_CALL)
    ]
    assert "15/15 valid" in output
    assert "Truncated: checked only the first 15 unique URLs; 3 omitted." in output
    assert all(url not in output for url in urls[MAX_URLS_PER_CALL:])


@pytest.mark.asyncio
async def test_connections_use_pinned_addresses_with_original_tls_identity(
    resolver, monkeypatch
):
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

        def get_extra_info(self, info):
            return None

        async def start_tls(self, ssl_context, server_hostname=None, timeout=None):
            assert ssl_context.check_hostname
            assert ssl_context.verify_mode == ssl.CERT_REQUIRED
            tls_hosts.append(server_hostname)
            return self

    async def connect_tcp(host, port, **kwargs):
        connections.append((host, port))
        return NetworkStream()

    monkeypatch.setattr(
        "httpcore._backends.auto.AutoBackend.connect_tcp",
        AsyncMock(side_effect=connect_tcp),
    )
    for variable in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
        monkeypatch.setenv(variable, "http://127.0.0.1:9999")
    results = await _check_urls_async(
        ["https://example.com/", "https://api.smith.langchain.com/info"], 1.0
    )
    assert all(result.valid for result in results)
    assert connections == [("93.184.216.34", 443), ("93.184.216.34", 443)]
    assert set(tls_hosts) == {"example.com", "api.smith.langchain.com"}
    assert any(b"Host: example.com\r\n" in buffer for buffer in writes)
    assert any(b"Host: api.smith.langchain.com\r\n" in buffer for buffer in writes)
