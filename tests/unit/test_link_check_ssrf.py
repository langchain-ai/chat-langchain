"""Test link-check destination restrictions with controlled DNS and HTTP."""

import socket
from unittest.mock import Mock

import httpx
import pytest

from src.tools import link_check_tools


@pytest.fixture(autouse=True)
def public_dns(monkeypatch):
    link_check_tools._cache.clear()
    resolver = Mock(
        return_value=[
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 0)),
        ]
    )
    monkeypatch.setattr(socket, "getaddrinfo", resolver)
    yield resolver
    link_check_tools._cache.clear()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://metadata.google.internal/",
        "https://webhook.site/callback",
        "http://[::1]/",
        "https://docs.langchain.com.attacker.example/",
        "https://attackerlangchain.com/",
        "https://docs.langchain.com@webhook.site/",
    ],
)
async def test_disallowed_urls_make_no_requests(url, monkeypatch, public_dns):
    handler = Mock(side_effect=AssertionError("Unexpected outbound request"))
    client_factory = httpx.AsyncClient
    monkeypatch.setattr(
        link_check_tools.httpx,
        "AsyncClient",
        lambda **kwargs: client_factory(
            transport=httpx.MockTransport(handler), **kwargs
        ),
    )
    result = await link_check_tools.check_links.ainvoke({"urls": [url]})
    assert "Invalid links:" in result
    assert f"{url}: host not allowed" in result
    handler.assert_not_called()
    public_dns.assert_not_called()
    assert url not in link_check_tools._cache


@pytest.mark.parametrize(
    "url",
    [
        "https://docs.langchain.com/",
        "https://DOCS.LANGCHAIN.COM/",
        "https://sub.docs.langchain.com/",
        "https://github.com/langchain-ai/langchain",
        "https://langchain-ai.github.io/",
    ],
)
def test_allowlisted_hosts(url):
    assert link_check_tools._is_allowed_host(url)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.1",
        "169.254.169.254",
        "192.168.1.1",
        "100.64.0.1",
        "240.0.0.1",
        "224.0.0.1",
        "0.0.0.0",
        "::1",
        "fe80::1",
        "fc00::1",
        "ff02::1",
        "::",
        "::ffff:127.0.0.1",
        "not-an-address",
    ],
)
async def test_non_public_dns_makes_no_requests(address, public_dns):
    public_dns.return_value.append(
        (
            socket.AF_INET6 if ":" in address else socket.AF_INET,
            socket.SOCK_STREAM,
            6,
            "",
            (address, 0),
        ),
    )
    handler = Mock(side_effect=AssertionError("Unexpected outbound request"))
    url = "https://docs.langchain.com/"
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await link_check_tools._check_single_url(client, url, 1.0)
    assert result.error == "host not allowed"
    handler.assert_not_called()
    assert url not in link_check_tools._cache


@pytest.mark.asyncio
@pytest.mark.parametrize("host", ["docs.langchain.com", "metadata.google.internal"])
@pytest.mark.parametrize("resolution", [[], socket.gaierror("DNS failed")])
async def test_dns_failure_is_rejected(host, resolution, public_dns):
    if isinstance(resolution, Exception):
        public_dns.side_effect = resolution
    else:
        public_dns.return_value = resolution
    handler = Mock(side_effect=AssertionError("Unexpected outbound request"))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await link_check_tools._check_single_url(
            client, f"https://{host}/", 1.0
        )
    assert result.error == "host not allowed"
    handler.assert_not_called()


@pytest.mark.asyncio
async def test_public_documentation_url_is_checked(monkeypatch, public_dns):
    handler = Mock(return_value=httpx.Response(200, text="Documentation"))
    client_factory = httpx.AsyncClient
    monkeypatch.setattr(
        link_check_tools.httpx,
        "AsyncClient",
        lambda **kwargs: client_factory(
            transport=httpx.MockTransport(handler), **kwargs
        ),
    )
    result = await link_check_tools.check_links.ainvoke(
        {"urls": ["https://docs.langchain.com/"]}
    )
    assert "1/1 valid" in result
    assert handler.call_count == 1
    public_dns.assert_called_once_with(
        "docs.langchain.com", None, type=socket.SOCK_STREAM
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source", ["https://docs.langchain.com/", "https://github.com/"]
)
@pytest.mark.parametrize(
    "target",
    [
        "https://webhook.site/callback",
        "http://127.0.0.1/",
        "http://169.254.169.254/",
        "ftp://docs.langchain.com/file",
    ],
)
async def test_unsafe_redirect_is_not_followed(source, target):
    handler = Mock(return_value=httpx.Response(302, headers={"Location": target}))
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        follow_redirects=True,
    ) as client:
        result = await link_check_tools._check_single_url(client, source, 1.0)
    assert result.error == "host not allowed"
    assert handler.call_count == 1
    assert str(handler.call_args.args[0].url) == source
    assert source not in link_check_tools._cache


@pytest.mark.asyncio
async def test_redirect_to_non_public_dns_is_not_followed(public_dns):
    public_dns.side_effect = [
        public_dns.return_value,
        [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 0)),
        ],
    ]
    handler = Mock(
        return_value=httpx.Response(
            302, headers={"Location": "https://support.langchain.com/"}
        )
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await link_check_tools._check_single_url(
            client, "https://docs.langchain.com/", 1.0
        )
    assert result.error == "host not allowed"
    assert handler.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("hops", [1, 5, 6])
async def test_redirect_limit(hops, public_dns):
    requests = []

    def handler(request):
        requests.append(request)
        position = int(request.url.path.strip("/"))
        if position < hops:
            target = (
                f"https://support.langchain.com/{position + 1}"
                if position == 0
                else f"/{position + 1}"
            )
            return httpx.Response(302, headers={"Location": target})
        return httpx.Response(200, text="Documentation")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await link_check_tools._check_single_url(
            client, "https://docs.langchain.com/0", 1.0
        )
    assert len(requests) == min(hops + 1, 6)
    assert public_dns.call_count == len(requests)
    if hops > 5:
        assert result.error == "Too many redirects"
    else:
        assert result.valid
        assert result.final_url == f"https://support.langchain.com/{hops}"


@pytest.mark.asyncio
async def test_head_fallback_retains_validation():
    handler = Mock(
        side_effect=[
            httpx.Response(405),
            httpx.Response(302, headers={"Location": "http://127.0.0.1/"}),
        ]
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await link_check_tools._check_single_url(
            client, "https://github.com/", 1.0
        )
    assert result.error == "host not allowed"
    assert [call.args[0].method for call in handler.call_args_list] == ["HEAD", "GET"]


@pytest.mark.asyncio
async def test_cached_success_does_not_bypass_dns(public_dns):
    url = "https://docs.langchain.com/"
    link_check_tools._cache[url] = link_check_tools.LinkCheckResult(url=url, valid=True)
    public_dns.side_effect = socket.gaierror("DNS failed")
    handler = Mock(side_effect=AssertionError("Unexpected outbound request"))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await link_check_tools._check_single_url(client, url, 1.0)
    assert result.error == "host not allowed"
    handler.assert_not_called()
