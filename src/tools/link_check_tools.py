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
ALLOWED_HOST_SUFFIXES = (
    "langchain.com",
    "github.com",
    "langchain-ai.github.io",
    "pypi.org",
    "npmjs.com",
)
USER_AGENT = "LangChain-LinkChecker/1.0"
CONTENT_CHECK_BYTES = 8192  # Only read first 8KB for soft 404 detection

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
    """Check for an HTTP URL with an allowed, non-literal hostname."""
    try:
        parsed = httpx.URL(url)
        hostname = parsed.host
        if parsed.scheme not in ("http", "https") or "." not in hostname:
            return False
        try:
            ipaddress.ip_address(hostname)
        except ValueError:
            return any(
                hostname == suffix or hostname.endswith("." + suffix)
                for suffix in ALLOWED_HOST_SUFFIXES
            )
        return False
    except (ValueError, httpx.InvalidURL):
        return False


class _HostNotAllowed(Exception):
    """Signal a destination that cannot be safely requested."""


class _ValidatedTransport(httpx.AsyncBaseTransport):
    """Pin connections to public addresses while preserving the HTTP/TLS host."""

    def __init__(self) -> None:
        self._transports: dict[tuple[str, str], httpx.AsyncHTTPTransport] = {}

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Validate all DNS answers and connect only to a validated address."""
        if not _is_valid_url(str(request.url)):
            raise _HostNotAllowed
        hostname = request.url.host
        try:
            addresses = await asyncio.wait_for(
                asyncio.get_running_loop().getaddrinfo(
                    hostname, request.url.port, type=socket.SOCK_STREAM
                ),
                timeout=request.extensions.get("timeout", {}).get("connect"),
            )
            if not addresses:
                raise _HostNotAllowed
            for address_info in addresses:
                destination = ipaddress.ip_address(address_info[4][0])
                if (
                    not destination.is_global
                    or destination.is_private
                    or destination.is_loopback
                    or destination.is_link_local
                    or destination.is_reserved
                    or destination.is_multicast
                    or destination.is_unspecified
                ):
                    raise _HostNotAllowed
        except (OSError, ValueError, TimeoutError) as exc:
            raise _HostNotAllowed from exc

        address = str(addresses[0][4][0])
        key = (hostname, address)
        if key not in self._transports:
            self._transports[key] = httpx.AsyncHTTPTransport(trust_env=False)
        pinned_request = httpx.Request(
            request.method,
            request.url.copy_with(host=address),
            headers=request.headers,
            stream=request.stream,
            extensions={**request.extensions, "sni_hostname": hostname},
        )
        return await self._transports[key].handle_async_request(pinned_request)

    async def aclose(self) -> None:
        """Close each hostname-isolated connection pool."""
        for transport in self._transports.values():
            await transport.aclose()


def _needs_soft_404_check(url: str) -> bool:
    """Check if URL is from a domain known to have soft 404s."""
    try:
        domain = urlparse(url).netloc.lower()
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
        for redirect_count in range(MAX_REDIRECTS + 1):
            if not _is_valid_url(current_url):
                return LinkCheckResult(url=url, valid=False, error="Host not allowed")
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

                if response.status_code in (
                    301,
                    302,
                    303,
                    307,
                    308,
                ) and response.headers.get("location"):
                    try:
                        current_url = urljoin(current_url, response.headers["location"])
                    except ValueError as exc:
                        raise _HostNotAllowed from exc
                    if not _is_valid_url(current_url):
                        return LinkCheckResult(
                            url=url, valid=False, error="Host not allowed"
                        )
                    if redirect_count == MAX_REDIRECTS:
                        raise httpx.TooManyRedirects("Too many redirects")
                    continue

                final_url = current_url if current_url != url else None
                is_valid = 200 <= response.status_code < 400
                error = None if is_valid else f"HTTP {response.status_code}"
                if needs_content_check and response.status_code == 200:
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
    except httpx.TimeoutException:
        result = LinkCheckResult(url=url, valid=False, error="Request timed out")
    except httpx.TooManyRedirects:
        result = LinkCheckResult(url=url, valid=False, error="Too many redirects")
    except httpx.ConnectError as e:
        result = LinkCheckResult(
            url=url, valid=False, error=f"Connection failed: {str(e)[:50]}"
        )
    except Exception as e:
        logger.warning(f"Error checking URL {url}: {e}")
        result = LinkCheckResult(url=url, valid=False, error=f"Error: {str(e)[:50]}")

    _cache[url] = result
    return result


async def _check_urls_async(urls: list[str], timeout: float) -> list[LinkCheckResult]:
    """Check multiple URLs concurrently."""
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
        transport=_ValidatedTransport(),
        trust_env=False,
        follow_redirects=False,
        max_redirects=MAX_REDIRECTS,
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
    """Check if URLs are valid and accessible before including them in a response.

    Args:
        urls: List of URLs to validate.
        timeout: Timeout per request in seconds (default: 10).

    Returns:
        Formatted results showing which URLs are valid/invalid with details.
    """
    if not urls:
        return "No URLs provided to check."

    # Deduplicate while preserving order
    seen = set()
    unique_urls = [u for u in urls if not (u in seen or seen.add(u))]

    results = await _check_urls_async(unique_urls, timeout)
    return _format_results(results)
