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
USER_AGENT = "LangChain-LinkChecker/1.0"
CONTENT_CHECK_BYTES = 8192  # Only read first 8KB for soft 404 detection

ALLOWED_HOSTS = {
    "docs.langchain.com",
    "langchain.com",
    "langchain.dev",
    "github.com",
    "support.langchain.com",
    "reference.langchain.com",
}

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


async def _is_allowed_url(url: str) -> bool:
    """Require a documentation host whose resolved addresses are all public."""
    try:
        parsed = httpx.URL(url)
        host = parsed.host
        if parsed.port is not None and not 1 <= parsed.port <= 65535:
            return False
        if parsed.scheme not in ("http", "https") or parsed.username or parsed.password:
            return False
        if not any(
            host == allowed or host.endswith(f".{allowed}") for allowed in ALLOWED_HOSTS
        ):
            return False
        addresses = await asyncio.wait_for(
            asyncio.get_running_loop().getaddrinfo(
                host,
                parsed.port or (443 if parsed.scheme == "https" else 80),
                type=socket.SOCK_STREAM,
            ),
            timeout=DEFAULT_TIMEOUT,
        )
        resolved_ips = [ipaddress.ip_address(address[4][0]) for address in addresses]
        return bool(resolved_ips) and all(
            address.is_global and not address.is_multicast and not address.is_reserved
            for address in resolved_ips
        )
    except (httpx.InvalidURL, ValueError, OSError, TimeoutError):
        return False


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
    if not await _is_allowed_url(url):
        return LinkCheckResult(
            url=url, valid=False, error="Host not allowed for link checking"
        )

    if url in _cache:
        return _cache[url]

    try:
        needs_content_check = _needs_soft_404_check(url)

        if needs_content_check:
            # Stream response, only read first chunk for soft 404 detection
            async with client.stream(
                "GET", url, timeout=timeout, follow_redirects=False
            ) as response:
                final_url = str(response.url) if str(response.url) != url else None
                is_valid = 200 <= response.status_code < 300

                if is_valid and response.status_code == 200:
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
        else:
            # Use HEAD for non-langchain domains (much faster)
            response = await client.head(url, timeout=timeout, follow_redirects=False)

            # Some servers don't support HEAD, fall back to GET
            if response.status_code == 405:
                if not await _is_allowed_url(url):
                    return LinkCheckResult(
                        url=url, valid=False, error="Host not allowed for link checking"
                    )
                response = await client.get(
                    url, timeout=timeout, follow_redirects=False
                )

            final_url = str(response.url) if str(response.url) != url else None
            is_valid = 200 <= response.status_code < 300

            result = LinkCheckResult(
                url=url,
                valid=is_valid,
                status_code=response.status_code,
                final_url=final_url,
                error=None if is_valid else f"HTTP {response.status_code}",
            )

        if 300 <= response.status_code < 400:
            location = response.headers.get("location")
            target = urljoin(str(response.url), location) if location else None
            if target and not await _is_allowed_url(target):
                return LinkCheckResult(
                    url=url, valid=False, error="Host not allowed for link checking"
                )
            return LinkCheckResult(
                url=url,
                valid=bool(target),
                status_code=response.status_code,
                final_url=target,
                error=None if target else "Redirect missing Location",
            )

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
    """Validate allowlisted documentation URLs to cite, not user-provided external or internal links."""
    if not urls:
        return "No URLs provided to check."

    # Deduplicate while preserving order
    seen = set()
    unique_urls = [u for u in urls if not (u in seen or seen.add(u))]

    results = await _check_urls_async(unique_urls, timeout)
    return _format_results(results)
