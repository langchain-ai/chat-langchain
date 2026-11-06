"""Link validation tool for checking URL validity before including in responses."""

import asyncio
import ipaddress
import logging
import math
import re
import socket
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
from langchain.tools import tool

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10.0
MAX_REDIRECTS = 5
MAX_URLS = 50
MAX_CONCURRENCY = 5
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
            and result.port in (None, 80, 443)
        )
    except Exception:
        return False


async def _public_address(url: str) -> str:
    """Resolve a URL only if every destination address is public."""
    if not _is_valid_url(url):
        raise ValueError("Invalid URL or disallowed port/credentials")
    parsed = httpx.URL(url)
    try:
        addresses = [ipaddress.ip_address(parsed.host)]
    except ValueError:
        records = await asyncio.get_running_loop().getaddrinfo(
            parsed.host,
            parsed.port or (443 if parsed.scheme == "https" else 80),
            type=socket.SOCK_STREAM,
        )
        addresses = [ipaddress.ip_address(record[4][0]) for record in records]
    if not addresses or any(
        not address.is_global
        or address.is_multicast
        or address.is_reserved
        or getattr(address, "ipv4_mapped", None) is not None
        for address in addresses
    ):
        raise ValueError("Destination is not a public address")
    return str(addresses[0])


class _PublicTransport(httpx.AsyncBaseTransport):
    """Pin each request, including redirects, to a validated public IP."""

    def __init__(self):
        # IP-based pooling must not reuse TLS connections across hostnames.
        self._transport = httpx.AsyncHTTPTransport(
            limits=httpx.Limits(
                max_connections=MAX_CONCURRENCY, max_keepalive_connections=0
            ),
            trust_env=False,
        )

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Validate and pin the destination while retaining Host and TLS identity."""
        address = await _public_address(str(request.url))
        pinned = httpx.Request(
            request.method,
            request.url.copy_with(host=address),
            headers=request.headers,
            stream=request.stream,
            extensions={**request.extensions, "sni_hostname": request.url.host},
        )
        return await self._transport.handle_async_request(pinned)

    async def aclose(self) -> None:
        """Close the underlying connection pool."""
        await self._transport.aclose()


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
    """Apply destination policy before cache access within a total time limit."""
    if not math.isfinite(timeout) or not 0 < timeout <= DEFAULT_TIMEOUT:
        return LinkCheckResult(
            url=url, valid=False, error="Timeout must be between 0 and 10 seconds"
        )
    try:
        async with asyncio.timeout(timeout):
            await _public_address(url)
            return await _check_single_url_impl(client, url, timeout)
    except TimeoutError:
        return LinkCheckResult(url=url, valid=False, error="Request timed out")
    except (ValueError, OSError):
        return LinkCheckResult(
            url=url,
            valid=False,
            error="URL rejected: destination must be public HTTP(S) on port 80 or 443",
        )


async def _check_single_url_impl(
    client: httpx.AsyncClient,
    url: str,
    timeout: float,
) -> LinkCheckResult:
    """Check a single URL for validity."""
    if url in _cache:
        return _cache[url]

    try:
        needs_content_check = _needs_soft_404_check(url)

        if needs_content_check:
            # Stream response, only read first chunk for soft 404 detection
            async with client.stream(
                "GET", url, timeout=timeout, follow_redirects=True
            ) as response:
                final_url = str(response.url) if str(response.url) != url else None
                is_valid = 200 <= response.status_code < 400

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
            response = await client.head(url, timeout=timeout, follow_redirects=True)

            # Some servers don't support HEAD, fall back to GET
            if response.status_code == 405:
                response = await client.get(url, timeout=timeout, follow_redirects=True)

            final_url = str(response.url) if str(response.url) != url else None
            is_valid = 200 <= response.status_code < 400

            result = LinkCheckResult(
                url=url,
                valid=is_valid,
                status_code=response.status_code,
                final_url=final_url,
                error=None if is_valid else f"HTTP {response.status_code}",
            )

        _cache[url] = result
        return result

    except ValueError:
        return LinkCheckResult(
            url=url,
            valid=False,
            error="URL rejected: destination must be public HTTP(S) on port 80 or 443",
        )
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
    if len(urls) > MAX_URLS:
        return [
            LinkCheckResult(
                url=url, valid=False, error=f"Batch exceeds {MAX_URLS} URL limit"
            )
            for url in urls
        ]
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async def check(url: str) -> LinkCheckResult:
        async with semaphore:
            return await _check_single_url(client, url, timeout)

    async with httpx.AsyncClient(
        transport=_PublicTransport(),
        trust_env=False,
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
        max_redirects=MAX_REDIRECTS,
    ) as client:
        tasks = [check(url) for url in urls]
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
    """Check up to 50 public HTTP(S) URLs on ports 80/443 with a 0–10 second timeout."""
    if len(urls) > MAX_URLS:
        return f"Batch exceeds {MAX_URLS} URL limit; no URLs checked."

    if not urls:
        return "No URLs provided to check."

    # Deduplicate while preserving order
    seen = set()
    unique_urls = [u for u in urls if not (u in seen or seen.add(u))]

    results = await _check_urls_async(unique_urls, timeout)
    return _format_results(results)
