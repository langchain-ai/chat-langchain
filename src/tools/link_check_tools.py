"""Link validation tool for checking URL validity before including in responses."""

import asyncio
import ipaddress
import logging
import os
import re
import socket
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import httpx
from langchain.tools import tool

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10.0
MAX_REDIRECTS = 5
ALLOWED_HOST_SUFFIXES = ("langchain.com",)
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
        return bool(
            result.scheme in ("http", "https")
            and result.hostname
            and not result.username
            and not result.password
            and httpx.URL(url).host == result.hostname
            and (result.port is None or 0 < result.port <= 65535)
        )
    except Exception:
        return False


def _is_allowed_host(url: str) -> bool:
    """Allow LangChain domains and explicitly configured documentation hosts."""
    hostname = urlparse(url).hostname
    if not hostname:
        return False
    extra_hosts = {
        host.strip().lower()
        for host in os.environ.get("CHECK_LINKS_ALLOWED_HOSTS", "").split(",")
        if host.strip()
    }
    return hostname in extra_hosts or any(
        hostname == suffix or hostname.endswith("." + suffix)
        for suffix in ALLOWED_HOST_SUFFIXES
    )


async def _resolve_destination(url: str, timeout: float) -> httpx.URL | None:
    """Resolve and pin a destination only when every address is public."""
    parsed = httpx.URL(url)
    try:
        addresses = await asyncio.wait_for(
            asyncio.to_thread(
                socket.getaddrinfo,
                parsed.host,
                parsed.port or (443 if parsed.scheme == "https" else 80),
                type=socket.SOCK_STREAM,
            ),
            timeout=timeout,
        )
        if not addresses:
            return None
        for address in addresses:
            ip = ipaddress.ip_address(address[4][0])
            if not ip.is_global or ip.is_multicast:
                return None
        return parsed.copy_with(host=addresses[0][4][0])
    except (OSError, ValueError):
        return None


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
        result = LinkCheckResult(url=url, valid=False, error="Invalid URL format")
        _cache[url] = result
        return result

    if not _is_allowed_host(url):
        return LinkCheckResult(url=url, valid=False, error="Host not allowed")

    if url in _cache:
        return _cache[url]

    try:
        current_url = url
        redirects = 0
        method = "GET" if _needs_soft_404_check(url) else "HEAD"
        while True:
            if not _is_valid_url(current_url):
                result = LinkCheckResult(
                    url=url, valid=False, error="Invalid URL format"
                )
                break
            if not _is_allowed_host(current_url):
                result = LinkCheckResult(url=url, valid=False, error="Host not allowed")
                break
            destination = await _resolve_destination(current_url, timeout)
            if destination is None:
                result = LinkCheckResult(url=url, valid=False, error="Host not allowed")
                break
            original = httpx.URL(current_url)
            async with client.stream(
                method,
                destination,
                timeout=timeout,
                follow_redirects=False,
                headers={"Host": original.netloc.decode("ascii")},
                extensions={"sni_hostname": original.host},
            ) as response:
                if response.status_code == 405 and method == "HEAD":
                    method = "GET"
                    continue
                if response.is_redirect:
                    if redirects >= MAX_REDIRECTS:
                        result = LinkCheckResult(
                            url=url, valid=False, error="Too many redirects"
                        )
                        break
                    current_url = urljoin(current_url, response.headers["location"])
                    redirects += 1
                    if _needs_soft_404_check(current_url):
                        method = "GET"
                    continue

                final_url = current_url if current_url != url else None
                is_valid = 200 <= response.status_code < 300
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
                break

        _cache[url] = result
        return result

    except (TimeoutError, httpx.TimeoutException):
        result = LinkCheckResult(url=url, valid=False, error="Request timed out")
    except httpx.TooManyRedirects:
        result = LinkCheckResult(url=url, valid=False, error="Too many redirects")
    except httpx.ConnectError:
        result = LinkCheckResult(url=url, valid=False, error="Connection failed")
    except Exception:
        logger.warning("Error checking documentation link")
        result = LinkCheckResult(url=url, valid=False, error="Link check failed")

    _cache[url] = result
    return result


async def _check_urls_async(urls: list[str], timeout: float) -> list[LinkCheckResult]:
    """Check multiple URLs concurrently."""
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
        follow_redirects=False,
        trust_env=False,
        # Do not reuse IP-keyed TLS connections across different hostnames.
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
    """Validate the agent's documentation citations on approved hosts, not user-supplied URLs."""
    if not urls:
        return "No URLs provided to check."

    # Deduplicate while preserving order
    seen = set()
    unique_urls = [u for u in urls if not (u in seen or seen.add(u))]

    results = await _check_urls_async(unique_urls, timeout)
    return _format_results(results)
