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


class _HostNotAllowed(ValueError):
    """Indicate that a destination cannot be safely requested."""


async def _resolve_destination(url: str, timeout: float) -> httpx.URL:
    """Resolve a public hostname and pin the URL to a validated address."""
    try:
        destination = httpx.URL(url)
        hostname = destination.host.rstrip(".")
        if destination.scheme not in ("http", "https") or "." not in hostname:
            raise _HostNotAllowed
        try:
            ipaddress.ip_address(hostname)
        except ValueError:
            pass
        else:
            raise _HostNotAllowed
        addresses = await asyncio.wait_for(
            asyncio.get_running_loop().getaddrinfo(
                hostname,
                destination.port or (443 if destination.scheme == "https" else 80),
                type=socket.SOCK_STREAM,
            ),
            timeout=timeout,
        )
        resolved = [ipaddress.ip_address(address[4][0]) for address in addresses]
        if not resolved or any(
            not address.is_global or address.is_multicast or address.is_reserved
            for address in resolved
        ):
            raise _HostNotAllowed
        return destination.copy_with(host=addresses[0][4][0])
    except (TimeoutError, ValueError, OSError, httpx.InvalidURL) as exc:
        raise _HostNotAllowed from exc


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
    # Check cache first
    if url in _cache:
        return _cache[url]

    if not _is_valid_url(url):
        result = LinkCheckResult(url=url, valid=False, error="Invalid URL format")
        _cache[url] = result
        return result

    try:
        current_url = url
        method = "GET" if _needs_soft_404_check(url) else "HEAD"
        redirects = 0
        while True:
            destination = await _resolve_destination(current_url, timeout)
            original = httpx.URL(current_url)
            async with client.stream(
                method,
                destination,
                headers={"Host": original.netloc.decode("ascii")},
                extensions={"sni_hostname": original.raw_host.decode("ascii")},
                timeout=timeout,
                follow_redirects=False,
            ) as response:
                location = response.headers.get("location")
                if response.status_code in (301, 302, 303, 307, 308) and location:
                    if redirects >= MAX_REDIRECTS:
                        raise httpx.TooManyRedirects("Too many redirects")
                    current_url = urljoin(current_url, location)
                    redirects += 1
                    if _needs_soft_404_check(current_url):
                        method = "GET"
                    continue
                if response.status_code == 405 and method == "HEAD":
                    method = "GET"
                    continue

                final_url = current_url if current_url != url else None
                is_valid = 200 <= response.status_code < 400

                if response.status_code == 200 and _needs_soft_404_check(current_url):
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
                if response.status_code < 500 and response.status_code not in (
                    408,
                    429,
                ):
                    _cache[url] = result
                return result

    except _HostNotAllowed:
        result = LinkCheckResult(url=url, valid=False, error="Host not allowed")
    except httpx.TimeoutException:
        result = LinkCheckResult(url=url, valid=False, error="Request timed out")
    except httpx.TooManyRedirects:
        result = LinkCheckResult(url=url, valid=False, error="Too many redirects")
    except httpx.ConnectError:
        result = LinkCheckResult(url=url, valid=False, error="Connection failed")
    except Exception as e:
        logger.warning(f"Error checking URL {url}: {e}")
        result = LinkCheckResult(url=url, valid=False, error="Request failed")

    return result


async def _check_urls_async(urls: list[str], timeout: float) -> list[LinkCheckResult]:
    """Check multiple URLs concurrently."""
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
        follow_redirects=False,
        trust_env=False,
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
