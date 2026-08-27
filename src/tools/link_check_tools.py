"""Link validation tool for checking URL validity before including in responses."""

import asyncio
import logging
import re
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
from langchain.tools import tool
from langchain_core.runnables import RunnableConfig

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
_run_caches: dict[str, dict[str, "LinkCheckResult"]] = {}


@dataclass
class LinkCheckResult:
    """Result of checking a single URL."""
    url: str
    valid: bool
    status_code: int | None = None
    error: str | None = None
    final_url: str | None = None
    cached: bool = False


def _get_cache(config: RunnableConfig | None) -> dict[str, LinkCheckResult]:
    """Get the link cache for the current run or thread."""
    if config:
        configurable = config.get("configurable", {})
        metadata = config.get("metadata", {})
        run_key = configurable.get("thread_id") or metadata.get("run_id")
        if run_key:
            return _run_caches.setdefault(str(run_key), {})
    return _cache


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


def _is_soft_404(content: str) -> bool:
    """Detect soft 404 pages that return HTTP 200 but show 'not found' content."""
    if "Article Not Found" in content:
        return True

    title_match = re.search(r'<title>(.*?)</title>', content, re.IGNORECASE)
    if title_match:
        title = title_match.group(1).lower()
        if any(phrase in title for phrase in ['not found', '404', 'page not found']):
            return True
    return False


async def _check_single_url(
    client: httpx.AsyncClient,
    url: str,
    timeout: float,
    cache: dict[str, LinkCheckResult] | None = None,
) -> LinkCheckResult:
    """Check a single URL for validity."""
    cache = cache if cache is not None else _cache
    # Check cache first
    if url in cache:
        result = cache[url]
        return LinkCheckResult(**{**result.__dict__, "cached": True})

    if not _is_valid_url(url):
        result = LinkCheckResult(url=url, valid=False, error="Invalid URL format")
        cache[url] = result
        return result

    try:
        needs_content_check = _needs_soft_404_check(url)

        if needs_content_check:
            # Stream response, only read first chunk for soft 404 detection
            async with client.stream("GET", url, timeout=timeout, follow_redirects=True) as response:
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
                            url=url, valid=False, status_code=200, final_url=final_url,
                            error="Soft 404: Page shows 'not found' content",
                        )
                        cache[url] = result
                        return result

                result = LinkCheckResult(
                    url=url, valid=is_valid, status_code=response.status_code,
                    final_url=final_url, error=None if is_valid else f"HTTP {response.status_code}",
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
                url=url, valid=is_valid, status_code=response.status_code,
                final_url=final_url, error=None if is_valid else f"HTTP {response.status_code}",
            )

        cache[url] = result
        return result

    except httpx.TimeoutException:
        result = LinkCheckResult(url=url, valid=False, error="Request timed out")
    except httpx.TooManyRedirects:
        result = LinkCheckResult(url=url, valid=False, error="Too many redirects")
    except httpx.ConnectError as e:
        result = LinkCheckResult(url=url, valid=False, error=f"Connection failed: {str(e)[:50]}")
    except Exception as e:
        logger.warning(f"Error checking URL {url}: {e}")
        result = LinkCheckResult(url=url, valid=False, error=f"Error: {str(e)[:50]}")

    cache[url] = result
    return result


async def _check_urls_async(
    urls: list[str], timeout: float, cache: dict[str, LinkCheckResult] | None = None
) -> list[LinkCheckResult]:
    """Check multiple URLs concurrently."""
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
        max_redirects=MAX_REDIRECTS,
    ) as client:
        tasks = [_check_single_url(client, url, timeout, cache) for url in urls]
        return list(await asyncio.gather(*tasks))


def _format_results(results: list[LinkCheckResult]) -> str:
    """Format check results into readable output."""
    if not results:
        return "No URLs to check."

    valid = [r for r in results if r.valid and not r.cached]
    cached_valid = [r for r in results if r.valid and r.cached]
    invalid = [r for r in results if not r.valid and not r.cached]
    cached_invalid = [r for r in results if not r.valid and r.cached]

    lines = [f"Link Check Results: {len(valid) + len(cached_valid)}/{len(results)} valid\n"]

    if invalid:
        lines.append("Invalid links:")
        lines.extend(f"  - {r.url}: {r.error}" for r in invalid)
        lines.append("")

    if cached_invalid:
        lines.append("Already checked earlier in this run (cached):")
        lines.extend(f"  - {r.url}: {r.error}" for r in cached_invalid)
        lines.append("")

    if valid:
        lines.append("Valid links:")
        for r in valid:
            suffix = f" (→ {r.final_url})" if r.final_url else ""
            lines.append(f"  - {r.url}{suffix}")

    if cached_valid:
        if valid:
            lines.append("")
        lines.append("Already validated earlier in this run (cached):")
        lines.extend(f"  - {r.url}" for r in cached_valid)

    return "\n".join(lines)


@tool
async def check_links(
    urls: list[str], timeout: float = DEFAULT_TIMEOUT, *, config: RunnableConfig
) -> str:
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

    cache = _get_cache(config)
    results = await _check_urls_async(unique_urls, timeout, cache)
    return _format_results(results)
