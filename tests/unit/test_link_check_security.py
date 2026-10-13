"""Test documentation link checks without making real network requests."""

import socket
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.tools.link_check_tools import (
    ALLOWED_HOSTS,
    DEFAULT_TIMEOUT,
    MAX_REDIRECTS,
    _cache,
    _check_single_url,
    _check_urls_async,
    _is_valid_url,
    _resolves_to_public_ip,
    check_links,
)


@pytest.fixture(autouse=True)
def clear_cache():
    _cache.clear()
    yield
    _cache.clear()


@pytest.fixture
def public_dns():
    with patch(
        "src.tools.link_check_tools._resolves_to_public_ip",
        new_callable=AsyncMock,
        return_value=True,
    ) as resolver:
        yield resolver


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8080/",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/",
        "http://10.0.0.1/",
        "https://example.com/",
        "https://docs.langchain.com.example.com/",
        "https://evillangchain.com/",
        "https://github.com/other-owner/repo",
        "https://github.com/langchain-ai-other/repo",
        "ftp://docs.langchain.com/",
        "https://docs.langchain.com:8080/",
        "https://user:password@docs.langchain.com/",
        "not-a-url",
        "http://[invalid]/",
    ],
)
@pytest.mark.asyncio
async def test_rejected_urls_make_no_network_call(url):
    client = AsyncMock(spec=httpx.AsyncClient)
    with patch("src.tools.link_check_tools.socket.getaddrinfo") as resolver:
        result = await _check_single_url(client, url, DEFAULT_TIMEOUT)

    assert not result.valid
    assert result.error == "Not an allowed documentation host"
    assert result.status_code is None
    resolver.assert_not_called()
    client.head.assert_not_called()
    client.get.assert_not_called()
    client.stream.assert_not_called()


@pytest.mark.parametrize("host", sorted(ALLOWED_HOSTS) + ["nested.docs.langchain.com"])
def test_documentation_hosts_are_allowed(host):
    path = "/langchain-ai/langchain" if host == "github.com" else "/documentation"
    assert _is_valid_url(f"https://{host.upper()}{path}")


@pytest.mark.parametrize(
    "addresses, expected",
    [
        (["8.8.8.8"], True),
        (["2606:4700:4700::1111"], True),
        (["127.0.0.1"], False),
        (["169.254.169.254"], False),
        (["10.0.0.1"], False),
        (["172.16.0.1"], False),
        (["192.168.1.1"], False),
        (["::1"], False),
        (["fc00::1"], False),
        (["fe80::1"], False),
        (["224.0.0.1"], False),
        (["ff02::1"], False),
        (["240.0.0.1"], False),
        (["::ffff:127.0.0.1"], False),
        (["::ffff:224.0.0.1"], False),
        (["8.8.8.8", "10.0.0.1"], False),
        ([], False),
    ],
)
@pytest.mark.asyncio
async def test_dns_rejects_any_non_public_address(addresses, expected):
    records = [
        (
            socket.AF_INET6 if ":" in address else socket.AF_INET,
            socket.SOCK_STREAM,
            socket.IPPROTO_TCP,
            "",
            (address, 0),
        )
        for address in addresses
    ]
    with patch("src.tools.link_check_tools.socket.getaddrinfo", return_value=records):
        assert await _resolves_to_public_ip("docs.langchain.com") is expected


@pytest.mark.asyncio
async def test_dns_failure_is_rejected():
    with patch(
        "src.tools.link_check_tools.socket.getaddrinfo", side_effect=socket.gaierror
    ):
        assert not await _resolves_to_public_ip("docs.langchain.com")


@pytest.mark.asyncio
async def test_allowed_host_with_private_dns_makes_no_http_call():
    client = AsyncMock(spec=httpx.AsyncClient)
    records = [
        (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("10.0.0.1", 0))
    ]
    with patch("src.tools.link_check_tools.socket.getaddrinfo", return_value=records):
        result = await _check_single_url(
            client, "https://docs.langchain.com/", DEFAULT_TIMEOUT
        )

    assert not result.valid
    assert result.error == "Not an allowed documentation host"
    client.head.assert_not_called()
    client.get.assert_not_called()
    client.stream.assert_not_called()


@pytest.mark.asyncio
async def test_allowed_documentation_url(public_dns):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, text="<title>LangChain documentation</title>")

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await _check_single_url(client, "https://docs.langchain.com/", 1000)

    assert result.valid
    assert result.status_code == 200
    assert len(requests) == 1
    assert requests[0].extensions["timeout"]["connect"] == DEFAULT_TIMEOUT
    public_dns.assert_awaited_once_with("docs.langchain.com")


@pytest.mark.parametrize("origin", ["https://docs.langchain.com/", "https://pypi.org/"])
@pytest.mark.parametrize(
    "target",
    [
        "http://127.0.0.1/",
        "https://example.com/",
        "https://github.com/other-owner/repo",
    ],
)
@pytest.mark.asyncio
async def test_disallowed_redirect_is_not_requested(origin, target, public_dns):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(302, headers={"Location": target})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond), follow_redirects=True
    ) as client:
        result = await _check_single_url(client, origin, DEFAULT_TIMEOUT)

    assert not result.valid
    assert result.error == "Redirects outside allowed documentation hosts"
    assert result.status_code is None
    assert len(requests) == 1
    public_dns.assert_awaited_once()


@pytest.mark.asyncio
async def test_redirect_to_allowed_host_with_private_dns_is_not_requested(public_dns):
    public_dns.side_effect = [True, False]
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            302, headers={"Location": "https://support.langchain.com/article"}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await _check_single_url(
            client, "https://docs.langchain.com/", DEFAULT_TIMEOUT
        )

    assert not result.valid
    assert result.error == "Redirects outside allowed documentation hosts"
    assert len(requests) == 1
    assert public_dns.await_count == 2


@pytest.mark.asyncio
async def test_relative_redirect_rechecks_dns_and_detects_soft_404(public_dns):
    requests = []

    def respond(request):
        requests.append(request)
        if request.url.path == "/":
            return httpx.Response(302, headers={"Location": "/missing"})
        return httpx.Response(200, text="<title>Page not found</title>")

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await _check_single_url(
            client, "https://docs.langchain.com/", DEFAULT_TIMEOUT
        )

    assert not result.valid
    assert result.error == "Soft 404: Page shows 'not found' content"
    assert result.final_url == "https://docs.langchain.com/missing"
    assert len(requests) == 2
    assert public_dns.await_count == 2


@pytest.mark.asyncio
async def test_allowed_cross_host_redirect_switches_to_content_check(public_dns):
    requests = []

    def respond(request):
        requests.append(request)
        if request.url.host == "pypi.org":
            return httpx.Response(
                301, headers={"Location": "https://docs.langchain.com/article"}
            )
        return httpx.Response(200, text="<title>Documentation</title>")

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await _check_single_url(client, "https://pypi.org/", DEFAULT_TIMEOUT)

    assert result.valid
    assert result.final_url == "https://docs.langchain.com/article"
    assert [request.method for request in requests] == ["HEAD", "GET"]
    assert [call.args[0] for call in public_dns.await_args_list] == [
        "pypi.org",
        "docs.langchain.com",
    ]


@pytest.mark.asyncio
async def test_head_fallback_get_redirect_is_checked(public_dns):
    requests = []

    def respond(request):
        requests.append(request)
        if request.method == "HEAD":
            return httpx.Response(405)
        return httpx.Response(302, headers={"Location": "http://169.254.169.254/"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await _check_single_url(client, "https://pypi.org/", DEFAULT_TIMEOUT)

    assert not result.valid
    assert result.error == "Redirects outside allowed documentation hosts"
    assert [request.method for request in requests] == ["HEAD", "GET"]


@pytest.mark.asyncio
async def test_redirect_limit(public_dns):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(302, headers={"Location": "/loop"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await _check_single_url(
            client, "https://docs.langchain.com/", DEFAULT_TIMEOUT
        )

    assert not result.valid
    assert result.error == "Too many redirects"
    assert len(requests) == MAX_REDIRECTS + 1


@pytest.mark.parametrize(
    "exception, error",
    [
        (httpx.ConnectError("secret internal address"), "Connection failed"),
        (RuntimeError("secret internal address"), "Unable to check documentation link"),
        (httpx.ReadTimeout("secret internal address"), "Request timed out"),
    ],
)
@pytest.mark.asyncio
async def test_connection_errors_are_sanitized(exception, error, public_dns):
    def respond(request):
        raise exception

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await _check_single_url(
            client, "https://docs.langchain.com/", DEFAULT_TIMEOUT
        )

    assert not result.valid
    assert result.error == error
    assert "secret" not in result.error


@pytest.mark.asyncio
async def test_client_disables_redirects_and_environment_proxies():
    with patch("src.tools.link_check_tools.httpx.AsyncClient") as factory:
        assert await _check_urls_async([], DEFAULT_TIMEOUT) == []

    assert factory.call_args.kwargs["follow_redirects"] is False
    assert factory.call_args.kwargs["trust_env"] is False


@pytest.mark.parametrize("timeout", [1000, float("inf"), float("nan")])
@pytest.mark.asyncio
async def test_tool_timeout_is_capped(timeout):
    with patch(
        "src.tools.link_check_tools._check_urls_async",
        new_callable=AsyncMock,
        return_value=[],
    ) as checker:
        await check_links.ainvoke(
            {"urls": ["https://docs.langchain.com/"], "timeout": timeout}
        )

    checker.assert_awaited_once_with(["https://docs.langchain.com/"], DEFAULT_TIMEOUT)
