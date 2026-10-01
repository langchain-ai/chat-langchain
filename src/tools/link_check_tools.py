"""Link validation tool for checking URL validity before including in responses."""

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx
from langchain.tools import tool
from pydantic import BaseModel, ConfigDict

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


class CheckLinksInput(BaseModel):
    """Flexible input schema for the link checker."""

    model_config = ConfigDict(extra="allow")

    urls: Any = None
    timeout: float = DEFAULT_TIMEOUT


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


def _normalize_urls(raw: Any) -> list[str]:
    """Normalize flexible tool input into unique URLs."""
    normalized: list[str] = []

    def add(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, (list, dict)):
                add(parsed)
                return
            for match in re.findall(r"https?://[^\s<>\"']+", value):
                cleaned = match.strip("`*_[]{}<>")
                while cleaned and cleaned[-1] in ".,;:!?)]}`*_":
                    cleaned = cleaned[:-1]
                while cleaned and cleaned[0] in "([{":
                    cleaned = cleaned[1:]
                if cleaned:
                    normalized.append(cleaned)
            return
        if isinstance(value, list):
            for item in value:
                add(item)
            return
        if isinstance(value, dict):
            for key in ("urls", "valid_urls", "links", "url"):
                if key in value:
                    add(value[key])
                    return

    add(raw)
    return list(dict.fromkeys(normalized))


def _edit_distance_at_most_one(left: str, right: str) -> bool:
    """Check whether two strings differ by at most one edit."""
    if abs(len(left) - len(right)) > 1:
        return False
    differences = 0
    left_index = right_index = 0
    while left_index < len(left) and right_index < len(right):
        if left[left_index] == right[right_index]:
            left_index += 1
            right_index += 1
            continue
        differences += 1
        if differences > 1:
            return False
        if len(left) > len(right):
            left_index += 1
        elif len(right) > len(left):
            right_index += 1
        else:
            left_index += 1
            right_index += 1
    if left_index < len(left) or right_index < len(right):
        differences += 1
    return differences <= 1


def _suggest_corrected_url(result: LinkCheckResult) -> str | None:
    """Suggest a known domain for a near-miss failing URL."""
    is_connection_failure = result.error and any(
        marker in result.error.lower() for marker in ("connection failed", "dns")
    )
    if result.status_code != 404 and not is_connection_failure:
        return None
    parsed = urlparse(result.url)
    host = parsed.hostname
    if not host:
        return None
    for domain in SOFT_404_DOMAINS:
        if _edit_distance_at_most_one(host.lower(), domain):
            return parsed._replace(netloc=domain).geturl()
    return None


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
                        _cache[url] = result
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

        _cache[url] = result
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

    _cache[url] = result
    return result


async def _check_urls_async(urls: list[str], timeout: float) -> list[LinkCheckResult]:
    """Check multiple URLs concurrently."""
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
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
        for result in invalid:
            suggestion = _suggest_corrected_url(result)
            detail = result.error
            if suggestion:
                detail = f"{detail}; suggested URL: {suggestion}"
            lines.append(f"  - {result.url}: {detail}")
        lines.append("")

    if valid:
        lines.append("Valid links:")
        for r in valid:
            suffix = f" (→ {r.final_url})" if r.final_url else ""
            lines.append(f"  - {r.url}{suffix}")

    return "\n".join(lines)


@tool(args_schema=CheckLinksInput)
async def check_links(
    urls: Any = None,
    timeout: float = DEFAULT_TIMEOUT,
    **extra: Any,
) -> str:
    """Check if URLs are valid and accessible before including them in a response.

    Args:
        urls: A URL, a list of URLs, or text containing URLs.
        timeout: Timeout per request in seconds (default: 10).

    Returns:
        Formatted results showing which URLs are valid/invalid with details.
    """
    raw_input = dict(extra)
    if urls is not None:
        raw_input["urls"] = urls
    normalized_urls = _normalize_urls(raw_input)
    if not normalized_urls:
        return "No URLs provided to check."

    results = await _check_urls_async(normalized_urls, timeout)
    return _format_results(results)
