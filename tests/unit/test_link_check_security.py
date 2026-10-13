"""Test URL policy and redirects without DNS or HTTP network access."""

import asyncio
import socket
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.tools import link_check_tools


def _dns_addresses(*addresses):
    return [
        (
            socket.AF_INET6 if ":" in address else socket.AF_INET,
            socket.SOCK_STREAM,
            6,
            "",
            (address, 443),
        )
        for address in addresses
    ]


@pytest.fixture(autouse=True)
def clear_cache():
    link_check_tools._cache.clear()
    yield
    link_check_tools._cache.clear()


def _run_check(url, handler, addresses=("8.8.8.8",)):
    async def invoke():
        with patch.object(
            asyncio.get_running_loop(),
            "getaddrinfo",
            new=AsyncMock(return_value=_dns_addresses(*addresses)),
        ) as resolver:
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ) as client:
                result = await link_check_tools._check_single_url(
                    client, url, link_check_tools.DEFAULT_TIMEOUT
                )
            return result, resolver

    return asyncio.run(invoke())


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data/",
        "http://127.0.0.1:8080/",
        "http://docs.langchain.com/",
        "https://some-unknown-domain.example/",
        "https://webhook.site/",
        "https://127.0.0.1/",
        "https://[::1]/",
        "https://user:password@docs.langchain.com/",
        "https://@docs.langchain.com/",
        "https://docs.langchain.com:8080/",
        "https://docs.langchain.com:invalid/",
        "https://docs.langchain.com:/",
        "https://docs.langchain.com.attacker.example/",
        "https://docs.langchain.com./",
        "https://docs.langchain.com\\@127.0.0.1/",
        "https://docs.langchain.com\n/",
        "not-a-url",
    ],
)
def test_rejected_urls_do_not_resolve_or_request(url):
    handler = AsyncMock()
    result, resolver = _run_check(url, handler)

    assert not result.valid
    assert result.error == "Not checked: host not allowed"
    handler.assert_not_called()
    resolver.assert_not_called()


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "169.254.169.254",
        "10.0.0.1",
        "172.16.0.1",
        "192.168.0.1",
        "224.0.0.1",
        "0.0.0.0",
        "240.0.0.1",
        "100.64.0.1",
        "192.0.2.1",
        "::1",
        "fe80::1",
        "fc00::1",
        "ff02::1",
        "::",
        "2001:db8::1",
        "::ffff:127.0.0.1",
    ],
)
def test_any_nonpublic_dns_answer_blocks_request(address):
    handler = AsyncMock()
    result, resolver = _run_check(
        "https://docs.langchain.com/", handler, ("8.8.8.8", address)
    )

    assert not result.valid
    assert result.error == "Not checked: host not allowed"
    handler.assert_not_called()
    resolver.assert_awaited_once()


@pytest.mark.parametrize(
    "url",
    [
        "https://docs.langchain.com/oss/python/langchain/overview",
        "https://DOCS.LANGCHAIN.COM:443/overview",
        "https://github.com/langchain-ai/langchain",
    ],
)
def test_allowed_url_is_checked(url):
    handler = AsyncMock(return_value=httpx.Response(200, text="Documentation"))
    result, _ = _run_check(url, handler)

    assert result.valid
    assert result.status_code == 200
    handler.assert_awaited_once()
    request = handler.call_args.args[0]
    assert request.method == ("HEAD" if request.url.host == "github.com" else "GET")
    assert request.extensions["timeout"]["read"] == link_check_tools.DEFAULT_TIMEOUT


@pytest.mark.parametrize(
    "url,fallback",
    [
        ("https://docs.langchain.com/start", False),
        ("https://github.com/start", False),
        ("https://github.com/start", True),
    ],
)
@pytest.mark.parametrize(
    "target",
    [
        "http://169.254.169.254/latest/meta-data/",
        "http://127.0.0.1:8080/",
        "https://attacker.example/",
        "//attacker.example/",
    ],
)
def test_redirect_targets_are_checked_for_stream_head_and_fallback(
    url, fallback, target
):
    requests = []

    def handler(request):
        requests.append(request)
        if fallback and request.method == "HEAD":
            return httpx.Response(405)
        return httpx.Response(302, headers={"Location": target})

    result, _ = _run_check(url, handler)

    assert not result.valid
    assert result.error == "Not checked: host not allowed"
    assert len(requests) == (2 if fallback else 1)
    assert all(str(request.url) == url for request in requests)


def test_relative_and_allowed_cross_host_redirects_are_followed():
    requests = []

    def handler(request):
        requests.append(str(request.url))
        if request.url.path == "/start":
            return httpx.Response(302, headers={"Location": "/next"})
        if request.url.path == "/next":
            return httpx.Response(
                307, headers={"Location": "https://support.langchain.com/article"}
            )
        return httpx.Response(200, text="Useful article")

    result, resolver = _run_check("https://docs.langchain.com/start", handler)

    assert result.valid
    assert result.final_url == "https://support.langchain.com/article"
    assert requests == [
        "https://docs.langchain.com/start",
        "https://docs.langchain.com/next",
        result.final_url,
    ]
    assert resolver.await_count == 3


def test_redirect_loop_stops_at_five_hops():
    handler = AsyncMock(return_value=httpx.Response(302, headers={"Location": "/loop"}))
    result, _ = _run_check("https://github.com/loop", handler)

    assert not result.valid
    assert result.error == "Too many redirects"
    assert handler.await_count == link_check_tools.MAX_REDIRECTS + 1


def test_head_falls_back_to_get():
    requests = []

    def handler(request):
        requests.append(request.method)
        return httpx.Response(405 if request.method == "HEAD" else 200)

    result, _ = _run_check("https://pypi.org/project/langchain/", handler)

    assert result.valid
    assert requests == ["HEAD", "GET"]


@pytest.mark.parametrize(
    "answers", [[], socket.gaierror("DNS unavailable"), TimeoutError()]
)
def test_dns_failure_is_not_checked(answers):
    async def invoke():
        resolver = (
            AsyncMock(side_effect=answers)
            if isinstance(answers, Exception)
            else AsyncMock(return_value=answers)
        )
        handler = AsyncMock()
        with patch.object(asyncio.get_running_loop(), "getaddrinfo", new=resolver):
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ) as client:
                result = await link_check_tools._check_single_url(
                    client, "https://docs.langchain.com/", 10.0
                )
        assert result.error == "Not checked: host not allowed"
        handler.assert_not_called()

    asyncio.run(invoke())


def test_cached_result_does_not_bypass_dns_policy():
    url = "https://docs.langchain.com/"
    link_check_tools._cache[url] = link_check_tools.LinkCheckResult(url=url, valid=True)
    handler = AsyncMock()
    result, _ = _run_check(url, handler, ("127.0.0.1",))

    assert not result.valid
    handler.assert_not_called()


@pytest.mark.parametrize("fallback", [False, True])
def test_dns_is_rechecked_before_redirect_or_get_fallback(fallback):
    async def invoke():
        handler = AsyncMock(
            return_value=httpx.Response(
                405 if fallback else 302,
                headers={"Location": "https://docs.langchain.com/next"},
            )
        )
        resolver = AsyncMock(
            side_effect=[_dns_addresses("8.8.8.8"), _dns_addresses("127.0.0.1")]
        )
        with patch.object(asyncio.get_running_loop(), "getaddrinfo", new=resolver):
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ) as client:
                result = await link_check_tools._check_single_url(
                    client, "https://github.com/start", 10.0
                )
        assert not result.valid
        assert result.error == "Not checked: host not allowed"
        handler.assert_awaited_once()
        assert resolver.await_count == 2

    asyncio.run(invoke())


def test_redirect_to_soft_404_host_uses_get_and_checks_content():
    requests = []

    def handler(request):
        requests.append(request.method)
        if request.url.host == "github.com":
            return httpx.Response(
                302, headers={"Location": "https://docs.langchain.com/missing"}
            )
        return httpx.Response(200, text="<title>Page not found</title>")

    result, _ = _run_check("https://github.com/start", handler)

    assert not result.valid
    assert result.error == "Soft 404: Page shows 'not found' content"
    assert requests == ["HEAD", "GET"]


def test_tool_uses_fixed_timeout_and_disables_automatic_redirects():
    assert "timeout" not in link_check_tools.check_links.args

    async def invoke():
        with patch("src.tools.link_check_tools.httpx.AsyncClient") as client_type:
            results = await link_check_tools._check_urls_async(
                [], link_check_tools.DEFAULT_TIMEOUT
            )
            assert results == []
            assert client_type.call_args.kwargs["follow_redirects"] is False
            assert client_type.call_args.kwargs["trust_env"] is False
        with patch(
            "src.tools.link_check_tools._check_urls_async",
            new=AsyncMock(return_value=[]),
        ) as check_urls:
            await link_check_tools.check_links.ainvoke(
                {"urls": ["https://docs.langchain.com/"]}
            )
            check_urls.assert_awaited_once_with(
                ["https://docs.langchain.com/"], link_check_tools.DEFAULT_TIMEOUT
            )

    asyncio.run(invoke())
