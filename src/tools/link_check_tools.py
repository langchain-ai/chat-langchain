"""Link validation tool for checking URL validity before including in responses."""

import asyncio
import ipaddress
import logging
import posixpath
import re
import socket
from dataclasses import dataclass
from urllib.parse import unquote, urljoin, urlparse

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

ALLOWED_HOSTS = SOFT_404_DOMAINS | {
    "reference.langchain.com",
    "academy.langchain.com",
    "langchain.com",
    "www.langchain.com",
    "docs.smith.langchain.com",
    "github.com",
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


def _is_allowed_url(url: str) -> bool:
    """Check the citation host, port, and GitHub organization boundary."""
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            return False
        if parsed.username is not None or parsed.password is not None:
            return False
        if parsed.hostname == "metadata.google.internal":
            return False
        try:
            ipaddress.ip_address(parsed.hostname)
        except ValueError:
            pass
        else:
            return False
        if parsed.hostname not in ALLOWED_HOSTS:
            return False
        if parsed.port not in (None, 443 if parsed.scheme == "https" else 80):
            return False
        if parsed.hostname == "github.com":
            path = unquote(parsed.path)
            return "\\" not in path and posixpath.normpath(path).startswith(
                "/langchain-ai/"
            )
        return True
    except ValueError:
        return False


async def _has_public_addresses(url: str, timeout: float) -> bool:
    """Resolve a host off the event loop and reject non-public addresses."""
    parsed = urlparse(url)
    addresses = await asyncio.wait_for(
        asyncio.to_thread(
            socket.getaddrinfo,
            parsed.hostname,
            parsed.port or (443 if parsed.scheme == "https" else 80),
            type=socket.SOCK_STREAM,
        ),
        timeout=timeout,
    )
    for address in addresses:
        resolved = ipaddress.ip_address(address[4][0])
        if (
            not resolved.is_global
            or resolved.is_loopback
            or resolved.is_private
            or resolved.is_link_local
            or resolved.is_reserved
            or resolved.is_multicast
        ):
            return False
    return bool(addresses)


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
        for redirect_count in range(MAX_REDIRECTS + 1):
            if not _is_allowed_url(current_url) or not await _has_public_addresses(
                current_url, timeout
            ):
                result = LinkCheckResult(
                    url=url,
                    valid=False,
                    error="Redirect to disallowed host"
                    if redirect_count
                    else "Host not allowed",
                )
                break

            soft_404 = False
            if _needs_soft_404_check(current_url):
                async with client.stream(
                    "GET", current_url, timeout=timeout
                ) as response:
                    content = ""
                    if response.status_code == 200:
                        async for chunk in response.aiter_text():
                            content += chunk
                            if len(content) >= CONTENT_CHECK_BYTES:
                                break
                        soft_404 = _is_soft_404(content)
            else:
                response = await client.head(current_url, timeout=timeout)
                if response.status_code == 405:
                    response = await client.get(current_url, timeout=timeout)

            if response.status_code in (301, 302, 303, 307, 308):
                location = response.headers.get("location")
                if not location:
                    result = LinkCheckResult(
                        url=url, valid=False, error="Invalid redirect"
                    )
                    break
                if redirect_count == MAX_REDIRECTS:
                    result = LinkCheckResult(
                        url=url, valid=False, error="Too many redirects"
                    )
                    break
                current_url = urljoin(current_url, location)
                continue

            final_url = current_url if current_url != url else None
            is_valid = 200 <= response.status_code < 400
            result = LinkCheckResult(
                url=url,
                valid=is_valid and not soft_404,
                status_code=response.status_code,
                final_url=final_url,
                error="Soft 404: Page shows 'not found' content"
                if soft_404
                else (None if is_valid else f"HTTP {response.status_code}"),
            )
            break

        _cache[url] = result
        return result

    except (httpx.TimeoutException, TimeoutError):
        result = LinkCheckResult(url=url, valid=False, error="Request timed out")
    except httpx.TooManyRedirects:
        result = LinkCheckResult(url=url, valid=False, error="Too many redirects")
    except (httpx.ConnectError, socket.gaierror):
        result = LinkCheckResult(url=url, valid=False, error="Connection failed")
    except Exception as e:
        logger.warning(f"Error checking URL {url}: {e}")
        result = LinkCheckResult(url=url, valid=False, error="Request error")

    _cache[url] = result
    return result


async def _check_urls_async(urls: list[str], timeout: float) -> list[LinkCheckResult]:
    """Check multiple URLs concurrently."""
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
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
    """Validate allowed documentation/support citation URLs from search results, not user-supplied reachability targets."""
    if not urls:
        return "No URLs provided to check."

    # Deduplicate while preserving order
    seen = set()
    unique_urls = [u for u in urls if not (u in seen or seen.add(u))]

    results = await _check_urls_async(unique_urls, timeout)
    return _format_results(results)
