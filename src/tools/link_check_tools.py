"""Link validation tool for checking URL validity before including in responses."""

import asyncio
import ipaddress
import posixpath
import re
import socket
from dataclasses import dataclass
from urllib.parse import unquote, urljoin, urlparse

import httpx
from langchain.tools import tool

DEFAULT_TIMEOUT = 10.0
MAX_REDIRECTS = 5
USER_AGENT = "LangChain-LinkChecker/1.0"
CONTENT_CHECK_BYTES = 8192  # Only read first 8KB for soft 404 detection

# Domains known to have soft 404s (return 200 with "not found" content)
SOFT_404_DOMAINS = {
    "docs.langchain.com",
    "python.langchain.com",
    "js.langchain.com",
    "support.langchain.com",
}

ALLOWED_DOMAINS = SOFT_404_DOMAINS | {
    "langchain-ai.github.io",
    "github.com",
    "smith.langchain.com",
    "langchain.com",
    "www.langchain.com",
}
DOMAIN_NOT_ALLOWED = "not checked (domain not allowed)"
UNREACHABLE = "invalid/unreachable"

# Simple in-memory cache
_cache: dict[str, "LinkCheckResult"] = {}


@dataclass
class LinkCheckResult:
    """Result of checking a single URL."""

    url: str
    valid: bool
    status_code: int | None = None
    error: str | None = None
    final_url: str | None = None


def _is_valid_url(url: str) -> bool:
    """Check whether a URL is an allowed documentation destination."""
    try:
        if "\\" in url or any(ord(character) <= 32 for character in url):
            return False
        result = httpx.URL(url)
        if (
            result.scheme not in ("http", "https")
            or result.host not in ALLOWED_DOMAINS
            or result.userinfo
            or result.port not in (None, 80 if result.scheme == "http" else 443)
        ):
            return False
        if result.host == "github.com":
            path = posixpath.normpath(unquote(result.path))
            return path == "/langchain-ai" or path.startswith("/langchain-ai/")
        return True
    except (ValueError, httpx.InvalidURL):
        return False


class _SafeTransport(httpx.AsyncHTTPTransport):
    """Connect only to validated public addresses, preserving TLS host verification."""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Resolve and pin an allowed destination before opening a connection."""
        if not _is_valid_url(str(request.url)):
            raise ValueError(DOMAIN_NOT_ALLOWED)
        host = request.url.host
        port = request.url.port or (443 if request.url.scheme == "https" else 80)
        addresses = await asyncio.wait_for(
            asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM),
            timeout=request.extensions.get("timeout", {}).get(
                "connect", DEFAULT_TIMEOUT
            ),
        )
        if not addresses:
            raise ValueError(UNREACHABLE)
        for address in addresses:
            parsed_address = ipaddress.ip_address(address[4][0])
            if (
                parsed_address.is_private
                or parsed_address.is_loopback
                or parsed_address.is_link_local
                or parsed_address.is_reserved
                or parsed_address.is_multicast
                or parsed_address.is_unspecified
                or not parsed_address.is_global
            ):
                raise ValueError(UNREACHABLE)
        pinned_request = httpx.Request(
            request.method,
            request.url.copy_with(host=addresses[0][4][0]),
            headers=request.headers,
            stream=request.stream,
            extensions={**request.extensions, "sni_hostname": host},
        )
        return await super().handle_async_request(pinned_request)


def _needs_soft_404_check(url: str) -> bool:
    """Check if URL is from a domain known to have soft 404s."""
    try:
        domain = urlparse(url).hostname
        return domain in SOFT_404_DOMAINS
    except Exception:
        return False


def _is_soft_404(content: str) -> bool:
    """Detect soft 404 pages that return HTTP 200 but show 'not found' content."""
    if "Article Not Found" in content:
        return True

    title_match = re.search(r"<title>(.*?)</title>", content, re.IGNORECASE)
    if title_match:
        title = title_match.group(1).lower()
        if any(phrase in title for phrase in ["not found", "404", "page not found"]):
            return True
    return False


async def _check_single_url(
    client: httpx.AsyncClient,
    url: str,
    timeout: float,
) -> LinkCheckResult:
    """Check a single URL for validity."""
    if not _is_valid_url(url):
        return LinkCheckResult(url=url, valid=False, error=DOMAIN_NOT_ALLOWED)

    if url in _cache:
        return _cache[url]

    try:
        current_url = url
        for redirect_count in range(MAX_REDIRECTS + 1):
            if not _is_valid_url(current_url):
                return LinkCheckResult(url=url, valid=False, error=DOMAIN_NOT_ALLOWED)
            needs_content_check = _needs_soft_404_check(current_url)
            method = "GET" if needs_content_check else "HEAD"
            async with client.stream(
                method, current_url, timeout=timeout, follow_redirects=False
            ) as response:
                if response.status_code == 405 and method == "HEAD":
                    await response.aclose()
                    response = await client.get(
                        current_url, timeout=timeout, follow_redirects=False
                    )
                if response.status_code in (301, 302, 303, 307, 308):
                    location = response.headers.get("location")
                    if not location or redirect_count == MAX_REDIRECTS:
                        break
                    current_url = urljoin(current_url, location)
                    continue
                is_valid = 200 <= response.status_code < 300
                if needs_content_check and is_valid and response.status_code == 200:
                    content = ""
                    async for chunk in response.aiter_text():
                        content += chunk
                        if len(content) >= CONTENT_CHECK_BYTES:
                            break
                    is_valid = not _is_soft_404(content)
                result = LinkCheckResult(
                    url=url,
                    valid=is_valid,
                    status_code=response.status_code,
                    final_url=current_url if current_url != url else None,
                    error=None if is_valid else UNREACHABLE,
                )
                _cache[url] = result
                return result
    except Exception:
        pass

    result = LinkCheckResult(url=url, valid=False, error=UNREACHABLE)

    _cache[url] = result
    return result


async def _check_urls_async(urls: list[str], timeout: float) -> list[LinkCheckResult]:
    """Check multiple URLs concurrently."""
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
        follow_redirects=False,
        trust_env=False,
        transport=_SafeTransport(
            trust_env=False, limits=httpx.Limits(max_keepalive_connections=0)
        ),
    ) as client:
        tasks = [_check_single_url(client, url, timeout) for url in urls]
        return list(await asyncio.gather(*tasks))


def _format_results(results: list[LinkCheckResult]) -> str:
    """Format check results into readable output."""
    if not results:
        return "No URLs to check."

    valid = [r for r in results if r.valid]
    invalid = [r for r in results if not r.valid]

    lines = [f"Link Check Results: {len(valid)}/{len(results)} valid\n"]

    if invalid:
        lines.append("Invalid links:")
        lines.extend(
            f"  - {r.url}: {DOMAIN_NOT_ALLOWED if r.error == DOMAIN_NOT_ALLOWED else UNREACHABLE}"
            for r in invalid
        )
        lines.append("")

    if valid:
        lines.append("Valid links:")
        for r in valid:
            suffix = f" (→ {r.final_url})" if r.final_url else ""
            lines.append(f"  - {r.url}{suffix}")

    return "\n".join(lines)


@tool
async def check_links(urls: list[str], timeout: float = DEFAULT_TIMEOUT) -> str:
    """Validate allowed URLs from documentation/support tools, never user-supplied URLs."""
    if not urls:
        return "No URLs provided to check."

    # Deduplicate while preserving order
    unique_urls = list(dict.fromkeys(urls))

    results = await _check_urls_async(unique_urls, timeout)
    return _format_results(results)
