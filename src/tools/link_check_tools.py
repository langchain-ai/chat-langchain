"""Link validation tool for checking URL validity before including in responses."""

import asyncio
import logging
import re
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime

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


_URL_RE = re.compile(r'https?://[^\s<>"\[\]`，。；！？]+')
_MARKDOWN_LINK_RE = re.compile(
    r'\[([^\]\n]+)\]\((https?://(?:[^()\s]|\([^()\s]*\))+)(?:\s+"[^"]*")?\)'
)


def _response_urls(text: str) -> list[str]:
    """Extract HTTP URLs without surrounding prose punctuation."""
    urls = []
    for match in _URL_RE.finditer(text):
        url = match.group().rstrip(".,;:!?")
        while url.endswith(")") and url.count(")") > url.count("("):
            url = url[:-1]
        urls.append(url)
    return urls


def _rewrite_links(text: str, replacements: dict[str, str]) -> str:
    """Replace invalid links while retaining their labels and surrounding text."""
    text = _MARKDOWN_LINK_RE.sub(
        lambda m: m.group(1) if replacements.get(m.group(2)) == "" else m.group(),
        text,
    )
    text = re.sub(
        r"<(https?://[^<>\s]+)>",
        lambda m: "" if replacements.get(m.group(1)) == "" else m.group(),
        text,
    )

    def replace_url(match: re.Match) -> str:
        raw = match.group()
        url = _response_urls(raw)[0]
        return replacements.get(url, url) + raw[len(url) :]

    text = _URL_RE.sub(replace_url, text)
    return re.sub(r"^\s*(?:[-*+] |\d+[.)] )\s*$", "", text, flags=re.MULTILINE)


class LinkValidationMiddleware(AgentMiddleware):
    """Enforce link checks on final model responses."""

    def after_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        """Validate final links for synchronous agent calls."""
        return asyncio.run(self.aafter_model(state, runtime))

    async def aafter_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        """Validate final links using only the current turn's tool results."""
        messages = state.get("messages", [])
        if not messages or not isinstance(messages[-1], AIMessage):
            return None
        response = messages[-1]
        if response.tool_calls:
            return None
        urls = list(dict.fromkeys(_response_urls(response.text)))
        if not urls:
            return None

        checked: dict[str, bool] = {}
        for message in reversed(messages[:-1]):
            if isinstance(message, HumanMessage):
                break
            if not isinstance(message, ToolMessage) or message.name != "check_links":
                continue
            valid = None
            for line in message.text.splitlines():
                if line == "Valid links:":
                    valid = True
                elif line == "Invalid links:":
                    valid = False
                elif valid is not None and line.startswith("  - "):
                    url = (
                        line[4:].split(" (→ ", 1)[0]
                        if valid
                        else line[4:].split(": ", 1)[0]
                    )
                    checked.setdefault(url, valid)

        unchecked = [url for url in urls if url not in checked]
        if unchecked:
            results = await _check_urls_async(unchecked, DEFAULT_TIMEOUT)
            checked.update((result.url, result.valid) for result in results)

        replacements = {}
        for url in urls:
            if checked.get(url, False):
                continue
            parsed = urlparse(url)
            replacement = parsed._replace(netloc="docs.langchain.com").geturl()
            replacements[url] = (
                replacement
                if parsed.netloc == "docs.ai.com" and checked.get(replacement, False)
                else ""
            )
        if not replacements:
            return None

        content = response.content
        if isinstance(content, str):
            content = _rewrite_links(content, replacements)
        else:
            content = [
                _rewrite_links(block, replacements)
                if isinstance(block, str)
                else {**block, "text": _rewrite_links(block["text"], replacements)}
                if block.get("type") == "text" and isinstance(block.get("text"), str)
                else block
                for block in content
            ]
        return {"messages": [response.model_copy(update={"content": content})]}
