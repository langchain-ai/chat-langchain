"""Test documentation-link host, address, redirect, and error restrictions."""

import socket
import ssl
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from src.tools import link_check_tools


def _addresses(*addresses):
    return [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 443))
        for address in addresses
    ]


@pytest.fixture(autouse=True)
def reset_checks(monkeypatch):
    link_check_tools._cache.clear()
    monkeypatch.delenv("CHECK_LINKS_ALLOWED_HOSTS", raising=False)
    resolver = Mock(return_value=_addresses("93.184.216.34"))
    monkeypatch.setattr(socket, "getaddrinfo", resolver)
    yield resolver
    link_check_tools._cache.clear()


async def _check(handler, url="https://docs.langchain.com/guide"):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), follow_redirects=True
    ) as client:
        return await link_check_tools._check_single_url(client, url, 1.0)


@pytest.mark.parametrize(
    "url",
    [
        "https://oast.example/",
        "http://169.254.169.254/",
        "http://localhost/",
        "https://notlangchain.com/",
        "https://langchain.com.oast.example/",
        "http://[::1]/",
    ],
)
@pytest.mark.asyncio
async def test_disallowed_hosts_never_resolve_or_request(url, reset_checks):
    handler = Mock()
    result = await _check(handler, url)
    assert not result.valid
    assert result.error == "Host not allowed"
    reset_checks.assert_not_called()
    handler.assert_not_called()


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.1",
        "172.16.0.1",
        "192.168.0.1",
        "169.254.169.254",
        "100.64.0.1",
        "0.0.0.0",
        "224.0.0.1",
        "::1",
        "fc00::1",
        "fe80::1",
        "ff0e::1",
        "::ffff:127.0.0.1",
    ],
)
@pytest.mark.asyncio
async def test_any_nonpublic_address_blocks_request(address, reset_checks):
    reset_checks.return_value = _addresses("93.184.216.34", address)
    handler = Mock()
    result = await _check(handler)
    assert result.error == "Host not allowed"
    handler.assert_not_called()


@pytest.mark.asyncio
async def test_public_documentation_request_is_pinned(reset_checks):
    def handler(request):
        assert request.url.host == "93.184.216.34"
        assert request.headers["host"] == "docs.langchain.com"
        assert request.extensions["sni_hostname"] == "docs.langchain.com"
        assert request.method == "GET"
        return httpx.Response(200, text="<title>Documentation</title>")

    result = await _check(handler)
    assert result.valid
    assert result.final_url is None
    reset_checks.assert_called_once_with(
        "docs.langchain.com", 443, type=socket.SOCK_STREAM
    )


@pytest.mark.asyncio
async def test_transport_connects_to_validated_ip_with_original_tls_identity():
    stream = AsyncMock()
    stream.start_tls.return_value = stream
    stream.get_extra_info = Mock(return_value=None)
    stream.read.return_value = (
        b"HTTP/1.1 200 OK\r\nContent-Length: 13\r\n\r\nDocumentation"
    )
    backend = AsyncMock()
    backend.connect_tcp.return_value = stream
    transport = httpx.AsyncHTTPTransport(
        limits=httpx.Limits(max_keepalive_connections=0)
    )
    transport._pool._network_backend = backend
    async with httpx.AsyncClient(transport=transport, trust_env=False) as client:
        result = await link_check_tools._check_single_url(
            client, "https://docs.langchain.com/guide", 1.0
        )
    assert result.valid
    assert backend.connect_tcp.call_args.kwargs["host"] == "93.184.216.34"
    tls_options = stream.start_tls.call_args.kwargs
    assert tls_options["server_hostname"] == "docs.langchain.com"
    assert tls_options["ssl_context"].check_hostname
    assert tls_options["ssl_context"].verify_mode == ssl.CERT_REQUIRED


@pytest.mark.parametrize(
    "url",
    [
        "ftp://docs.langchain.com/",
        "https://user:pass@docs.langchain.com/",
        "https://docs.langchain.com:bad/",
    ],
)
@pytest.mark.asyncio
async def test_invalid_url_is_not_resolved_or_requested(url, reset_checks):
    handler = Mock()
    assert (await _check(handler, url)).error == "Invalid URL format"
    handler.assert_not_called()
    reset_checks.assert_not_called()


@pytest.mark.asyncio
async def test_check_links_tool_checks_allowed_citations(monkeypatch):
    factory = httpx.AsyncClient
    handler = Mock(return_value=httpx.Response(200, text="Documentation"))
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: factory(transport=httpx.MockTransport(handler), **kwargs),
    )
    result = await link_check_tools.check_links.ainvoke(
        {"urls": ["https://docs.langchain.com/guide", "https://oast.example/"]}
    )
    assert "1/2 valid" in result
    assert "https://oast.example/: Host not allowed" in result
    assert handler.call_count == 1


@pytest.mark.asyncio
async def test_allowed_ipv6_is_pinned(reset_checks):
    reset_checks.return_value = _addresses("2606:4700:4700::1111")
    handler = Mock(return_value=httpx.Response(200, text="Documentation"))
    assert (await _check(handler)).valid
    assert handler.call_args.args[0].url.host == "2606:4700:4700::1111"


def test_extra_hosts_are_exact_only(monkeypatch):
    monkeypatch.setenv("CHECK_LINKS_ALLOWED_HOSTS", " github.com, PYPI.ORG ")
    for hostname in ["github.com", "pypi.org", "langchain.com", "docs.langchain.com"]:
        assert link_check_tools._is_allowed_host("https://" + hostname)
    for hostname in ["api.github.com", "evilpypi.org", "github.com.evil.test"]:
        assert not link_check_tools._is_allowed_host("https://" + hostname)


@pytest.mark.parametrize(
    "location",
    ["https://oast.example/", "http://localhost/", "http://169.254.169.254/"],
)
@pytest.mark.asyncio
async def test_disallowed_redirect_is_not_requested(location, reset_checks):
    handler = Mock(return_value=httpx.Response(302, headers={"location": location}))
    result = await _check(handler)
    assert result.error == "Host not allowed"
    assert handler.call_count == 1
    assert reset_checks.call_count == 1


@pytest.mark.asyncio
async def test_redirect_revalidates_addresses(reset_checks):
    reset_checks.side_effect = [_addresses("93.184.216.34"), _addresses("10.0.0.1")]
    handler = Mock(return_value=httpx.Response(302, headers={"location": "/next"}))
    assert (await _check(handler)).error == "Host not allowed"
    assert handler.call_count == 1
    assert reset_checks.call_count == 2


@pytest.mark.asyncio
async def test_relative_redirect_preserves_logical_url():
    def handler(request):
        if request.url.path == "/guide":
            return httpx.Response(302, headers={"location": "next"})
        assert request.url.path == "/next"
        return httpx.Response(200, text="Documentation")

    result = await _check(handler)
    assert result.valid
    assert result.final_url == "https://docs.langchain.com/next"


@pytest.mark.asyncio
async def test_redirect_limit():
    handler = Mock(return_value=httpx.Response(302, headers={"location": "/loop"}))
    assert (await _check(handler)).error == "Too many redirects"
    assert handler.call_count == link_check_tools.MAX_REDIRECTS + 1


@pytest.mark.asyncio
async def test_head_fallback_revalidates_and_pins(reset_checks):
    def handler(request):
        assert request.url.host == "93.184.216.34"
        assert request.headers["host"] == "www.langchain.com"
        return httpx.Response(405 if request.method == "HEAD" else 200)

    assert (await _check(handler, "https://www.langchain.com/")).valid
    assert reset_checks.call_count == 2


@pytest.mark.asyncio
async def test_head_fallback_blocks_dns_rebinding(reset_checks):
    reset_checks.side_effect = [_addresses("93.184.216.34"), _addresses("127.0.0.1")]
    handler = Mock(return_value=httpx.Response(405))
    assert (
        await _check(handler, "https://www.langchain.com/")
    ).error == "Host not allowed"
    assert handler.call_count == 1


@pytest.mark.parametrize("resolution", [[], socket.gaierror("secret resolver details")])
@pytest.mark.asyncio
async def test_dns_failures_are_generic(resolution, reset_checks):
    if isinstance(resolution, Exception):
        reset_checks.side_effect = resolution
    else:
        reset_checks.return_value = resolution
    handler = Mock()
    assert (await _check(handler)).error == "Host not allowed"
    handler.assert_not_called()


@pytest.mark.parametrize(
    ("exception", "error"),
    [
        (httpx.ConnectError("secret connection details"), "Connection failed"),
        (httpx.ReadTimeout("secret timeout details"), "Request timed out"),
        (RuntimeError("secret implementation details"), "Link check failed"),
    ],
)
@pytest.mark.asyncio
async def test_network_errors_are_generic(exception, error):
    assert (await _check(Mock(side_effect=exception))).error == error


@pytest.mark.asyncio
async def test_soft_404_after_redirect():
    def handler(request):
        if request.headers["host"] == "www.langchain.com":
            return httpx.Response(
                302, headers={"location": "https://docs.langchain.com/missing"}
            )
        assert request.method == "GET"
        return httpx.Response(200, text="<title>Page not found</title>")

    result = await _check(handler, "https://www.langchain.com/")
    assert not result.valid
    assert result.error == "Soft 404: Page shows 'not found' content"


@pytest.mark.asyncio
async def test_client_disables_proxies_redirects_and_connection_reuse(monkeypatch):
    factory = Mock(wraps=httpx.AsyncClient)
    monkeypatch.setattr(httpx, "AsyncClient", factory)
    await link_check_tools._check_urls_async([], 1.0)
    options = factory.call_args.kwargs
    assert options["follow_redirects"] is False
    assert options["trust_env"] is False
    assert options["limits"].max_keepalive_connections == 0
