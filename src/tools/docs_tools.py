"""LangChain Documentation Search Tool with Fuzzy Cache Matching."""
# Tools:
#   - SearchDocsByLangChain

import json
import logging
import os
import time
from typing import Any

import requests
from langchain.tools import tool
from langsmith.run_helpers import get_current_run_tree
from rapidfuzz import fuzz

from src.tools.redis import RedisCache

logger = logging.getLogger(__name__)

# Configuration
MINTLIFY_API_URL = os.getenv("MINTLIFY_API_URL")
DEFAULT_PAGE_SIZE = 3
MAX_PAGE_SIZE = 10
MAX_RETRIES = 3
DEFAULT_LANGUAGE = "python"
CACHE_TTL_SECONDS = 86400
FUZZY_THRESHOLD = 75
CACHE_MAX_ENTRIES = int(os.getenv("CACHE_MAX_ENTRIES", "0"))
FUZZY_SCAN_LIMIT = int(os.getenv("FUZZY_SCAN_LIMIT", "300"))
FUZZY_SCAN_COUNT = int(os.getenv("FUZZY_SCAN_COUNT", "1000"))

# Metric keys stored in Redis
METRIC_HITS_EXACT = "metrics:hits_exact"
METRIC_HITS_FUZZY = "metrics:hits_fuzzy"
METRIC_MISSES = "metrics:misses"
METRIC_API_REQUESTS = "metrics:api_requests_total"

ABBREVIATIONS = {
    "auth": "authentication",
    "config": "configuration",
    "configs": "configuration",
    "deploy": "deployment",
}

_cache = RedisCache(ttl_seconds=CACHE_TTL_SECONDS, max_entries=CACHE_MAX_ENTRIES)


def _normalize_query(query: str) -> str:
    """Normalize query: lowercase, expand abbreviations."""
    words = query.lower().strip().split()
    return " ".join(ABBREVIATIONS.get(w, w) for w in words)


def _increment_metric(key: str) -> None:
    try:
        _cache.incr(key)
    except Exception:
        pass


def _find_fuzzy_match(
    normalized_query: str,
    page_size: int,
    language: str,
) -> tuple[str | None, float, int]:
    """Scan cache for best fuzzy match. Returns (key, score, keys_scanned)."""
    first_word = normalized_query.split()[0] if normalized_query else ""
    pattern = f"{first_word}*|{page_size}|{language}" if first_word else f"*|{page_size}|{language}"

    best_key = None
    best_score = 0.0
    cursor = 0
    keys_scanned = 0

    while True:
        cursor, keys = _cache.scan(cursor, match=pattern, count=FUZZY_SCAN_COUNT)

        for cached_key in keys:
            keys_scanned += 1
            if FUZZY_SCAN_LIMIT > 0 and keys_scanned > FUZZY_SCAN_LIMIT:
                break

            parts = cached_key.split("|")
            if len(parts) != 3:
                continue

            cached_query, cached_size, cached_lang = parts
            if cached_size != str(page_size) or cached_lang != language:
                continue

            score = fuzz.token_set_ratio(normalized_query, cached_query)
            if score > best_score:
                best_score = score
                best_key = cached_key

        if cursor == 0 or (FUZZY_SCAN_LIMIT > 0 and keys_scanned > FUZZY_SCAN_LIMIT):
            break

    return best_key, best_score, keys_scanned


def _get_from_cache_fuzzy(
    query: str,
    page_size: int,
    language: str = DEFAULT_LANGUAGE,
) -> str | None:
    """Get result from cache with exact match, then fuzzy fallback."""
    normalized = _normalize_query(query)
    exact_key = f"{normalized}|{page_size}|{language}"

    # Try exact match first
    result = _cache.get(exact_key)
    if result:
        _increment_metric(METRIC_HITS_EXACT)
        logger.debug(f"Cache hit (exact): '{query}'")
        return result

    # Try fuzzy match
    best_key, best_score, _ = _find_fuzzy_match(normalized, page_size, language)

    if best_score >= FUZZY_THRESHOLD and best_key:
        result = _cache.get(best_key)
        if result:
            _increment_metric(METRIC_HITS_FUZZY)
            logger.debug(f"Cache hit (fuzzy {best_score:.0f}%): '{query}'")
            _cache.set(exact_key, result)  # Store for future exact match
            return result

    _increment_metric(METRIC_MISSES)
    return None


def _store_in_cache(query: str, page_size: int, language: str, result: str) -> None:
    """Store search result in cache."""
    key = f"{_normalize_query(query)}|{page_size}|{language}"
    _cache.set(key, result)


def get_cache_stats() -> dict[str, Any]:
    """Get cache performance statistics."""
    stats = _cache.stats()

    try:
        hits_exact = int(_cache.get(METRIC_HITS_EXACT) or "0")
        hits_fuzzy = int(_cache.get(METRIC_HITS_FUZZY) or "0")
        misses = int(_cache.get(METRIC_MISSES) or "0")
        api_requests = int(_cache.get(METRIC_API_REQUESTS) or "0")
    except Exception:
        hits_exact = hits_fuzzy = misses = api_requests = 0

    total = hits_exact + hits_fuzzy + misses
    hit_rate = ((hits_exact + hits_fuzzy) / total * 100) if total > 0 else 0

    return {
        **stats,
        "hits_exact": hits_exact,
        "hits_fuzzy": hits_fuzzy,
        "misses": misses,
        "total_requests": total,
        "hit_rate_percent": round(hit_rate, 1),
        "api_requests_total": api_requests,
        "fuzzy_threshold": FUZZY_THRESHOLD,
    }


def clear_cache() -> None:
    """Clear search cache and reset metrics."""
    _cache.clear()
    for key in [METRIC_HITS_EXACT, METRIC_HITS_FUZZY, METRIC_MISSES, METRIC_API_REQUESTS]:
        try:
            _cache.set(key, "0", ttl_seconds=315360000)
        except Exception:
            pass


def _get_api_key() -> str:
    if not MINTLIFY_API_URL:
        raise ValueError("MINTLIFY_API_URL not configured")
    api_key = os.getenv("MINTLIFY_API_KEY")
    if not api_key:
        raise ValueError("MINTLIFY_API_KEY not found in environment")
    return api_key


def _track_docs_for_langsmith(urls: list[str]) -> None:
    """Track retrieved doc URLs in LangSmith run metadata."""
    if not urls:
        return
    try:
        run_tree = get_current_run_tree()
        if not run_tree:
            return

        existing = run_tree.metadata.get("retrieved_docs", []) if run_tree.metadata else []
        all_docs = (existing if isinstance(existing, list) else []) + urls

        unique = []
        seen = set()
        for doc in all_docs:
            if doc not in seen:
                seen.add(doc)
                unique.append(doc)

        run_tree.add_metadata({"retrieved_docs": unique})
    except Exception:
        pass


def _format_search_results(results: list[dict[str, Any]]) -> str:
    """Format search results into readable text."""
    if not results:
        return "No results found."

    formatted = []
    urls = []

    for i, result in enumerate(results, 1):
        metadata = result.get("metadata", {})
        title = metadata.get("title", "Untitled")
        path = result.get("path", "")
        content = result.get("content", "")
        url = f"https://docs.langchain.com{path}" if path else "N/A"

        if path:
            urls.append(url)

        formatted.append(
            f"Result {i}:\n"
            f"Title: {title}\n"
            f"Link: {url}\n"
            f"Content: {content}\n"
        )

    _track_docs_for_langsmith(urls)
    return "\n---\n\n".join(formatted)


def _search_docs_api(
    query: str,
    page_size: int = DEFAULT_PAGE_SIZE,
    version: str | None = None,
    language: str | None = None,
) -> str:
    """Execute documentation search via Mintlify API."""
    headers = {
        "Authorization": f"Bearer {_get_api_key()}",
        "Content-Type": "application/json",
    }

    payload: dict[str, Any] = {"query": query, "pageSize": page_size}
    if version or language:
        payload["filter"] = {}
        if version:
            payload["filter"]["version"] = version
        if language:
            payload["filter"]["language"] = language

    _increment_metric(METRIC_API_REQUESTS)

    response = requests.post(MINTLIFY_API_URL, json=payload, headers=headers, timeout=30)
    response.raise_for_status()

    data = response.json()
    if not isinstance(data, list):
        if isinstance(data, dict) and (error := data.get("error") or data.get("message")):
            raise ValueError(f"Mintlify API error: {error}")
        data = []

    return _format_search_results(data)


@tool
def SearchDocsByLangChain(
    query: str,
    page_size: int = DEFAULT_PAGE_SIZE,
    version: str | None = None,
    language: str = DEFAULT_LANGUAGE,
) -> str:
    """Search LangChain, LangGraph, and LangSmith documentation.

    Args:
        query: Natural language search query.
        page_size: Number of results (default: 3, max: 10).
        version: Optional version filter (e.g., "v1", "latest").
        language: Programming language filter (default: "python").

    Returns:
        Formatted results with title, link, and content for each match.
    """
    page_size = max(1, min(page_size, MAX_PAGE_SIZE))

    cached = _get_from_cache_fuzzy(query, page_size, language)
    if cached is not None:
        return cached

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = _search_docs_api(query, page_size, version, language)
            _store_in_cache(query, page_size, language, result)
            return result
        except Exception as e:
            last_error = e
            logger.warning(f"Docs search attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(0.5)

    logger.error(f"Docs search failed after {MAX_RETRIES} attempts: {last_error}")
    return json.dumps({
        "error": "Documentation search unavailable",
        "message": f"Search failed after {MAX_RETRIES} attempts.",
        "query": query,
        "suggestion": "Check https://docs.langchain.com directly.",
        "details": str(last_error)[:100],
    })
