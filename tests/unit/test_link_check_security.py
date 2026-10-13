"""Tests for the shared link checker network policy."""

import asyncio
import socket
from unittest.mock import Mock

import httpx
import pytest

from src.tools import link_check_tools


@pytest.fixture
def network(monkeypatch):
    link_check_tools._cache.clear()
    resolver = Mock(
        return_value=[
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ]
    )
    monkeypatch.setattr(link_check_tools.socket, "getaddrinfo", resolver)
    requests = []
    responses = {}
    settings = []

    def handle(request):
        requests.append(request)
        status, headers, text = responses.get(
            (request.method, str(request.url)), (200, {}, "Documentation")
        )
        return httpx.Response(status, headers=headers, text=text)

    original_client = httpx.AsyncClient

    def client(**kwargs):
        settings.append(kwargs)
        return original_client(transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setattr(link_check_tools.httpx, "AsyncClient", client)
    yield requests, responses, resolver, settings
    link_check_tools._cache.clear()


def check_url(url):
    return asyncio.run(link_check_tools._check_urls_async([url], 1.0))[0]


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://metadata.google.internal/computeMetadata/v1/",
        "https://example.com/",
        "http://[::1]/",
        "https://docs.langchain.com.attacker.example/",
        "https://github.com/other-org/repo",
        "https://github.com/langchain-ai-evil/repo",
        "https://github.com/langchain-ai/../other-org/repo",
        "https://github.com/langchain-ai/%2e%2e/other-org/repo",
        "https://docs.langchain.com:8080/",
        "https://user:password@docs.langchain.com/",
    ],
)
def test_blocked_hosts_do_not_resolve_or_request(network, url):
    requests, _, resolver, _ = network
    result = check_url(url)

    assert not result.valid
    assert result.error == "Host not allowed"
    assert requests == []
    resolver.assert_not_called()


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.1",
        "169.254.169.254",
        "100.100.100.200",
        "168.63.129.16",
        "224.0.0.1",
        "240.0.0.1",
        "::1",
        "fe80::1",
        "fc00::1",
        "ff02::1",
        "::ffff:127.0.0.1",
    ],
)
def test_allowed_hostname_with_unsafe_dns_is_rejected(network, address):
    requests, _, resolver, _ = network
    resolver.return_value.append(
        (
            socket.AF_INET6 if ":" in address else socket.AF_INET,
            1,
            6,
            "",
            (address, 443),
        )
    )

    result = check_url("https://docs.langchain.com/guide")

    assert not result.valid
    assert result.error == "Host not allowed"
    assert requests == []


@pytest.mark.parametrize("addresses", [[], socket.gaierror("No DNS answer")])
def test_dns_failures_are_closed(network, addresses):
    requests, _, resolver, _ = network
    if isinstance(addresses, Exception):
        resolver.side_effect = addresses
    else:
        resolver.return_value = addresses

    result = check_url("https://docs.langchain.com/guide")

    assert result.error == "Host not allowed"
    assert requests == []


@pytest.mark.parametrize(
    "host", sorted(link_check_tools.ALLOWED_HOSTS) + ["github.com/langchain-ai/repo"]
)
def test_allowed_documentation_is_checked(network, host):
    requests, _, resolver, settings = network
    url = f"https://{host}/guide"

    result = check_url(url)

    assert result.valid
    assert result.status_code == 200
    assert len(requests) == 1
    assert str(requests[0].url) == url
    resolver.assert_called_once()
    assert settings[0]["follow_redirects"] is False
    assert settings[0]["trust_env"] is False


@pytest.mark.parametrize(
    "source,method",
    [
        ("https://docs.langchain.com/guide", "GET"),
        ("https://langchain.com/guide", "HEAD"),
    ],
)
@pytest.mark.parametrize(
    "location",
    [
        "http://127.0.0.1/admin",
        "//169.254.169.254/latest/meta-data/",
        "https://example.com/",
        "http://metadata.google.internal/",
        "https://github.com/other-org/repo",
    ],
)
def test_redirect_to_prohibited_host_is_not_requested(
    network, source, method, location
):
    requests, responses, resolver, _ = network
    responses[(method, source)] = (302, {"Location": location}, "")

    result = check_url(source)

    assert result.error == "Host not allowed"
    assert not result.valid
    assert [str(request.url) for request in requests] == [source]
    resolver.assert_called_once()


def test_redirect_dns_is_checked_again(network):
    requests, responses, resolver, _ = network
    source = "https://docs.langchain.com/guide"
    responses[("GET", source)] = (302, {"Location": "/next"}, "")
    resolver.side_effect = [
        resolver.return_value,
        [(socket.AF_INET, 1, 6, "", ("127.0.0.1", 443))],
    ]

    result = check_url(source)

    assert result.error == "Host not allowed"
    assert len(requests) == 1
    assert resolver.call_count == 2


def test_relative_redirect_and_soft_404_check(network):
    requests, responses, _, _ = network
    source = "https://docs.langchain.com/guide"
    destination = "https://docs.langchain.com/next"
    responses[("GET", source)] = (302, {"Location": "/next"}, "")
    responses[("GET", destination)] = (200, {}, "<title>Page not found</title>")

    result = check_url(source)

    assert not result.valid
    assert result.error == "Soft 404: Page shows 'not found' content"
    assert result.final_url == destination
    assert len(requests) == 2


def test_allowed_redirect_preserves_final_url(network):
    requests, responses, _, _ = network
    source = "https://langchain.com/guide"
    destination = "https://docs.langchain.com/guide"
    responses[("HEAD", source)] = (301, {"Location": destination}, "")

    result = check_url(source)

    assert result.valid
    assert result.final_url == destination
    assert [request.method for request in requests] == ["HEAD", "GET"]


@pytest.mark.parametrize("extra_redirect", [False, True])
def test_redirect_limit(network, extra_redirect):
    requests, responses, _, _ = network
    redirects = link_check_tools.MAX_REDIRECTS + int(extra_redirect)
    for index in range(redirects):
        responses[("GET", f"https://docs.langchain.com/{index}")] = (
            302,
            {"Location": f"/{index + 1}"},
            "",
        )

    result = check_url("https://docs.langchain.com/0")

    assert len(requests) == link_check_tools.MAX_REDIRECTS + 1
    assert result.valid is not extra_redirect
    assert result.error == ("Too many redirects" if extra_redirect else None)


def test_head_fallback_cannot_redirect_to_internal_host(network):
    requests, responses, _, _ = network
    source = "https://github.com/langchain-ai/langchain"
    responses[("HEAD", source)] = (405, {}, "")
    responses[("GET", source)] = (302, {"Location": "http://127.0.0.1/"}, "")

    result = check_url(source)

    assert result.error == "Host not allowed"
    assert [request.method for request in requests] == ["HEAD", "GET"]
    assert all(str(request.url) == source for request in requests)


def test_tool_uses_shared_security_policy(network):
    requests, _, _, _ = network
    result = asyncio.run(
        link_check_tools.check_links.ainvoke(
            {"urls": ["https://docs.langchain.com/guide", "http://127.0.0.1/"]}
        )
    )

    assert "1/2 valid" in result
    assert "http://127.0.0.1/: Host not allowed" in result
    assert [str(request.url) for request in requests] == [
        "https://docs.langchain.com/guide"
    ]
