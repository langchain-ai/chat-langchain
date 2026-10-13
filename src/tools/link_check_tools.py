"""Link validation tool for checking URL validity before including in responses."""

import asyncio
import ipaddress
import logging
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
ALLOWED_LINK_HOSTS = frozenset(
    {
        "docs.langchain.com",
        "python.langchain.com",
        "js.langchain.com",
        "support.langchain.com",
        "reference.langchain.com",
        "langchain.com",
        "www.langchain.com",
        "blog.langchain.com",
        "smith.langchain.com",
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
    """Check if a string is a valid URL format."""
    try:
        result = urlparse(url)
        return all([result.scheme in ("http", "https"), result.netloc])
    except Exception:
        return False


def _needs_soft_404_check(url: str) -> bool:
    """Check if URL is from a domain known to have soft 404s."""
    try:
        domain = urlparse(url).netloc.lower()
        return domain in SOFT_404_DOMAINS
    except Exception:
        return False


def _is_allowed_host(url: str) -> bool:
    """Allow only named documentation hosts and their subdomains."""
    try:
        host = urlparse(url).hostname
        if not host or not _is_valid_url(url):
            return False
        host = host.lower()
        if httpx.URL(url).host != host:
            return False
        try:
            ipaddress.ip_address(host)
        except ValueError:
            return any(
                host == allowed or host.endswith("." + allowed)
                for allowed in ALLOWED_LINK_HOSTS
            )
        return False
    except (ValueError, httpx.InvalidURL):
        return False


def _resolves_to_public_ip(host: str) -> bool:
    """Reject hosts unless every resolved address is public unicast."""
    try:
        addresses = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
        if not addresses:
            return False
        for address in addresses:
            parsed = ipaddress.ip_address(address[4][0])
            if (
                not parsed.is_global
                or parsed.is_private
                or parsed.is_loopback
                or parsed.is_link_local
                or parsed.is_reserved
                or parsed.is_multicast
                or parsed.is_unspecified
            ):
                return False
        return True
    except (OSError, ValueError):
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
        result = LinkCheckResult(url=url, valid=False, error="Invalid URL format")
        _cache[url] = result
        return result

    try:
        current_url = url
        for redirects in range(MAX_REDIRECTS + 1):
            if not _is_allowed_host(current_url) or not await asyncio.to_thread(
                _resolves_to_public_ip, httpx.URL(current_url).host
            ):
                return LinkCheckResult(url=url, valid=False, error="host not allowed")

            if redirects == 0 and url in _cache:
                return _cache[url]

            needs_content_check = _needs_soft_404_check(current_url)
            content = ""
            if needs_content_check:
                async with client.stream(
                    "GET", current_url, timeout=timeout, follow_redirects=False
                ) as response:
                    if response.status_code == 200:
                        async for chunk in response.aiter_text():
                            content += chunk
                            if len(content) >= CONTENT_CHECK_BYTES:
                                break
            else:
                response = await client.head(
                    current_url, timeout=timeout, follow_redirects=False
                )
                if response.status_code == 405:
                    response = await client.get(
                        current_url, timeout=timeout, follow_redirects=False
                    )

            if (
                response.status_code in (301, 302, 303, 307, 308)
                and "location" in response.headers
            ):
                if redirects == MAX_REDIRECTS:
                    raise httpx.TooManyRedirects("Too many redirects")
                current_url = str(
                    httpx.URL(current_url).join(response.headers["location"])
                )
                continue

            if (
                needs_content_check
                and response.status_code == 200
                and _is_soft_404(content)
            ):
                result = LinkCheckResult(
                    url=url,
                    valid=False,
                    status_code=200,
                    final_url=current_url if current_url != url else None,
                    error="Soft 404: Page shows 'not found' content",
                )
                _cache[url] = result
                return result

            final_url = str(response.url) if str(response.url) != url else None
            is_valid = 200 <= response.status_code < 400

            result = LinkCheckResult(
                url=url,
                valid=is_valid,
                status_code=response.status_code,
                final_url=final_url,
                error=None if is_valid else f"HTTP {response.status_code}",
            )
            break

        _cache[url] = result
        return result

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
    """Check allowlisted LangChain/GitHub documentation URLs with public DNS and safe redirects."""
    if not urls:
        return "No URLs provided to check."

    # Deduplicate while preserving order
    seen = set()
    unique_urls = [u for u in urls if not (u in seen or seen.add(u))]

    results = await _check_urls_async(unique_urls, timeout)
    return _format_results(results)
