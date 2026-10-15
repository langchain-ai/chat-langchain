"""Link validation tool for checking URL validity before including in responses."""

import asyncio
import ipaddress
import logging
import re
import socket
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import httpx
from langchain.tools import tool

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10.0
MAX_REDIRECTS = 5
USER_AGENT = "LangChain-LinkChecker/1.0"
CONTENT_CHECK_BYTES = 8192  # Only read first 8KB for soft 404 detection
ALLOWED_LINK_HOSTS = frozenset(
    {
        "docs.langchain.com",
        "python.langchain.com",
        "js.langchain.com",
        "reference.langchain.com",
        "support.langchain.com",
        "smith.langchain.com",
        "langchain.com",
        "www.langchain.com",
        "blog.langchain.com",
        "github.com",
        "langchain-ai.github.io",
    }
)

# Domains known to have soft 404s (return 200 with "not found" content)
SOFT_404_DOMAINS = {
    "docs.langchain.com",
    "python.langchain.com",
    "js.langchain.com",
    "support.langchain.com",
}

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
    """Check URL syntax and the destination allowlist without resolving DNS."""
    try:
        result = urlparse(url)
        hostname = result.hostname
        if result.scheme not in ("http", "https") or not hostname:
            return False
        if result.port not in (None, 443 if result.scheme == "https" else 80):
            return False
        if result.netloc.endswith(":") or len(hostname) > 253:
            return False
        if result.username is not None or result.password is not None:
            return False
        if any(ord(character) <= 32 for character in url):
            return False
        if hostname != httpx.URL(url).host or not all(
            re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label)
            for label in hostname.split(".")
        ):
            return False
        return any(
            hostname == allowed or hostname.endswith("." + allowed)
            for allowed in ALLOWED_LINK_HOSTS
        )
    except (ValueError, httpx.InvalidURL):
        return False


class _HostNotAllowed(ValueError):
    """Identify a blocked URL or resolved address."""


class _SafeLinkTransport(httpx.AsyncBaseTransport):
    """Pin requests to public addresses while retaining the original HTTP/TLS host."""

    def __init__(self) -> None:
        """Keep connection pools isolated by the original hostname identity."""
        self._transports: dict[tuple[str, str, int], httpx.AsyncHTTPTransport] = {}

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Validate DNS answers and connect only to a checked numeric address."""
        if not _is_valid_url(str(request.url)):
            raise _HostNotAllowed
        hostname = request.url.host
        port = request.url.port or (443 if request.url.scheme == "https" else 80)
        addresses = await asyncio.wait_for(
            asyncio.get_running_loop().getaddrinfo(
                hostname,
                port,
                type=socket.SOCK_STREAM,
            ),
            timeout=request.extensions.get("timeout", {}).get(
                "connect", DEFAULT_TIMEOUT
            ),
        )
        if not addresses:
            raise httpx.ConnectError("unreachable")
        resolved = [ipaddress.ip_address(address[4][0]) for address in addresses]
        if any(
            not address.is_global or address.is_multicast or address.is_reserved
            for address in resolved
        ):
            raise _HostNotAllowed
        pinned_request = httpx.Request(
            request.method,
            request.url.copy_with(host=str(resolved[0])),
            headers=request.headers,
            stream=request.stream,
            extensions={**request.extensions, "sni_hostname": hostname},
        )
        origin = (request.url.scheme, hostname, port)
        transport = self._transports.get(origin)
        if transport is None:
            transport = httpx.AsyncHTTPTransport(trust_env=False)
            self._transports[origin] = transport
        return await transport.handle_async_request(pinned_request)

    async def aclose(self) -> None:
        """Close all hostname-isolated connection pools."""
        await asyncio.gather(
            *(transport.aclose() for transport in self._transports.values())
        )


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
        return LinkCheckResult(url=url, valid=False, error="Host not allowed")

    if url in _cache:
        return _cache[url]

    try:
        current_url = url
        method = "GET" if _needs_soft_404_check(current_url) else "HEAD"
        redirects = 0
        while True:
            if not _is_valid_url(current_url):
                raise _HostNotAllowed
            async with client.stream(
                method, current_url, timeout=timeout, follow_redirects=False
            ) as response:
                if response.is_redirect and "location" in response.headers:
                    if redirects >= MAX_REDIRECTS:
                        return LinkCheckResult(
                            url=url, valid=False, error="Too many redirects"
                        )
                    current_url = urljoin(
                        str(response.url), response.headers["location"]
                    )
                    redirects += 1
                    method = "GET" if _needs_soft_404_check(current_url) else "HEAD"
                    continue
                if method == "HEAD" and response.status_code == 405:
                    method = "GET"
                    continue

                final_url = str(response.url) if str(response.url) != url else None
                is_valid = 200 <= response.status_code < 400
                error = None if is_valid else f"HTTP {response.status_code}"

                if _needs_soft_404_check(current_url) and response.status_code == 200:
                    content = ""
                    async for chunk in response.aiter_text():
                        content += chunk
                        if len(content) >= CONTENT_CHECK_BYTES:
                            break

                    if _is_soft_404(content):
                        is_valid = False
                        error = "Soft 404: Page shows 'not found' content"

                result = LinkCheckResult(
                    url=url,
                    valid=is_valid,
                    status_code=response.status_code,
                    final_url=final_url,
                    error=error,
                )
            break

        _cache[url] = result
        return result

    except _HostNotAllowed:
        return LinkCheckResult(url=url, valid=False, error="Host not allowed")
    except Exception:
        logger.warning("Link check unreachable")
        return LinkCheckResult(url=url, valid=False, error="unreachable")


async def _check_urls_async(urls: list[str], timeout: float) -> list[LinkCheckResult]:
    """Check multiple URLs concurrently."""
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
        follow_redirects=False,
        transport=_SafeLinkTransport(),
        trust_env=False,
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
        lines.extend(f"  - {r.url}: {r.error}" for r in invalid)
        lines.append("")

    if valid:
        lines.append("Valid links:")
        for r in valid:
            suffix = f" (→ {r.final_url})" if r.final_url else ""
            lines.append(f"  - {r.url}{suffix}")

    return "\n".join(lines)


@tool
async def check_links(urls: list[str], timeout: float = DEFAULT_TIMEOUT) -> str:
    """Check allowlisted LangChain and GitHub URLs before including them in a response."""
    if not urls:
        return "No URLs provided to check."

    # Deduplicate while preserving order
    seen = set()
    unique_urls = [u for u in urls if not (u in seen or seen.add(u))]

    results = await _check_urls_async(unique_urls, timeout)
    return _format_results(results)
