"""Link validation tool for checking URL validity before including in responses."""

import asyncio
import ipaddress
import logging
import posixpath
import re
import socket
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
from langchain.tools import tool

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10.0
MAX_REDIRECTS = 5
USER_AGENT = "LangChain-LinkChecker/1.0"
CONTENT_CHECK_BYTES = 8192  # Only read first 8KB for soft 404 detection
ALLOWED_HOST_SUFFIXES = ("langchain.com", "langchain.dev")

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
    """Check if a string is a valid URL format."""
    try:
        result = urlparse(url)
        return all([result.scheme in ("http", "https"), result.netloc])
    except Exception:
        return False


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


async def _validate_target(url: str) -> tuple[httpx.URL, str] | None:
    """Resolve an approved destination and reject any unsafe address."""
    try:
        target = httpx.URL(url)
        host = target.host
        allowed = any(
            host == suffix or host.endswith("." + suffix)
            for suffix in ALLOWED_HOST_SUFFIXES
        ) or (
            host == "github.com"
            and posixpath.normpath(target.path).startswith("/langchain-ai/")
        )
        if (
            target.scheme not in ("http", "https")
            or not allowed
            or target.userinfo
            or (target.port is not None and not 0 < target.port <= 65535)
        ):
            return None
        addresses = await asyncio.get_running_loop().getaddrinfo(
            host,
            target.port or (443 if target.scheme == "https" else 80),
            type=socket.SOCK_STREAM,
        )
        if not addresses:
            return None
        for address_info in addresses:
            address = ipaddress.ip_address(str(address_info[4][0]))
            if (
                not address.is_global
                or address.is_multicast
                or address.is_reserved
                or (
                    isinstance(address, ipaddress.IPv6Address) and address.is_site_local
                )
            ):
                return None
        return target, str(addresses[0][4][0])
    except (httpx.InvalidURL, OSError, ValueError, OverflowError):
        return None


async def _check_single_url(
    client: httpx.AsyncClient,
    url: str,
    timeout: float,
) -> LinkCheckResult:
    """Check a single URL for validity."""
    if not _is_valid_url(url):
        return LinkCheckResult(url=url, valid=False, error="Invalid URL format")

    target = await _validate_target(url)
    if target is None:
        return LinkCheckResult(
            url=url, valid=False, error="Host not allowed for link checking"
        )
    if url in _cache:
        cached = _cache[url]
        if cached.final_url and await _validate_target(cached.final_url) is None:
            return LinkCheckResult(
                url=url, valid=False, error="Host not allowed for link checking"
            )
        return cached

    try:
        current_url = url
        method = "GET" if _needs_soft_404_check(url) else "HEAD"
        redirects = 0
        while True:
            if target is None:
                return LinkCheckResult(
                    url=url, valid=False, error="Host not allowed for link checking"
                )
            destination, address = target
            async with client.stream(
                method,
                destination.copy_with(host=address),
                headers={"Host": destination.netloc.decode("ascii")},
                extensions={"sni_hostname": destination.host},
                timeout=timeout,
                follow_redirects=False,
            ) as response:
                if (
                    response.status_code in (301, 302, 303, 307, 308)
                    and "location" in response.headers
                ):
                    if redirects >= MAX_REDIRECTS:
                        raise httpx.TooManyRedirects("Too many redirects")
                    current_url = str(destination.join(response.headers["location"]))
                    redirects += 1
                    if _needs_soft_404_check(current_url):
                        method = "GET"
                elif response.status_code == 405 and method == "HEAD":
                    method = "GET"
                else:
                    final_url = current_url if current_url != url else None
                    is_valid = 200 <= response.status_code < 400
                    if (
                        is_valid
                        and response.status_code == 200
                        and _needs_soft_404_check(current_url)
                    ):
                        content = ""
                        async for chunk in response.aiter_text():
                            content += chunk
                            if len(content) >= CONTENT_CHECK_BYTES:
                                break
                        if _is_soft_404(content):
                            result = LinkCheckResult(
                                url=url,
                                valid=False,
                                status_code=200,
                                final_url=final_url,
                                error="Soft 404: Page shows 'not found' content",
                            )
                            _cache[url] = result
                            return result
                    result = LinkCheckResult(
                        url=url,
                        valid=is_valid,
                        status_code=response.status_code,
                        final_url=final_url,
                        error=None if is_valid else f"HTTP {response.status_code}",
                    )
                    _cache[url] = result
                    return result
            target = await _validate_target(current_url)

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
        follow_redirects=False,
        trust_env=False,
        # IP-pinned origins must not reuse TLS sessions across distinct hostnames.
        limits=httpx.Limits(max_keepalive_connections=0),
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
    """Check public HTTP(S) links on langchain.com, langchain.dev, their subdomains, or github.com/langchain-ai/ paths."""
    if not urls:
        return "No URLs provided to check."

    # Deduplicate while preserving order
    seen = set()
    unique_urls = [u for u in urls if not (u in seen or seen.add(u))]

    results = await _check_urls_async(unique_urls, timeout)
    return _format_results(results)
