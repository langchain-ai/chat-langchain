"""Link validation tool for checking URL validity before including in responses."""

import asyncio
import logging
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import unquote, urlparse

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


class _AnchorParser(HTMLParser):
    """Collect element anchors and heading text from an HTML page."""

    def __init__(self) -> None:
        super().__init__()
        self.anchors: set[str] = set()
        self._heading_parts: list[str] | None = None
        self.heading_texts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        for attribute in ("id", "name"):
            value = attributes.get(attribute)
            if value:
                self.anchors.add(value)
        if tag.lower() in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._heading_parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"h1", "h2", "h3", "h4", "h5", "h6"} and self._heading_parts is not None:
            self.heading_texts.append(" ".join(self._heading_parts))
            self._heading_parts = None

    def handle_data(self, data: str) -> None:
        if self._heading_parts is not None:
            self._heading_parts.append(data)


def _slugify_heading(text: str) -> str:
    """Convert heading text to the anchor slug used by docs pages."""
    text = re.sub(r"[^\w\s-]", "", text.lower())
    return re.sub(r"-+", "-", re.sub(r"\s+", "-", text)).strip("-")


def _anchor_exists(content: str, fragment: str) -> bool:
    """Check whether an HTML page contains the requested anchor."""
    parser = _AnchorParser()
    parser.feed(content)
    decoded_fragment = unquote(fragment)
    return decoded_fragment in parser.anchors or any(
        _slugify_heading(text) == decoded_fragment.lower() for text in parser.heading_texts
    )


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
        parsed_url = urlparse(url)
        fragment = parsed_url.fragment

        if fragment:
            async with client.stream("GET", url, timeout=timeout, follow_redirects=True) as response:
                final_url = str(response.url) if str(response.url) != url else None
                is_valid = 200 <= response.status_code < 400
                content = ""
                if is_valid:
                    async for chunk in response.aiter_text():
                        content += chunk

                    if response.status_code == 200 and _is_soft_404(content):
                        result = LinkCheckResult(
                            url=url, valid=False, status_code=200, final_url=final_url,
                            error="Soft 404: Page shows 'not found' content",
                        )
                        _cache[url] = result
                        return result

                    if not _anchor_exists(content, fragment):
                        result = LinkCheckResult(
                            url=url, valid=False, status_code=200, final_url=final_url,
                            error=f"anchor '#{fragment}' not found on page",
                        )
                        _cache[url] = result
                        return result

                result = LinkCheckResult(
                    url=url, valid=is_valid, status_code=response.status_code,
                    final_url=final_url, error=None if is_valid else f"HTTP {response.status_code}",
                )
            _cache[url] = result
            return result

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
