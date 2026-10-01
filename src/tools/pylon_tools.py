# Pylon Knowledge Base Tools
# Tools:
#   - search_support_articles
#   - get_support_article_content
import html
import json
import logging
import os
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv()

logger = logging.getLogger(__name__)

# Pylon API configuration
PYLON_API_BASE_URL = "https://api.usepylon.com"


class PylonUnavailableError(RuntimeError):
    """Raised when the Pylon knowledge base cannot be reached."""


def _get_kb_id() -> str:
    """Get knowledge base ID from environment."""
    kb_id = os.getenv("PYLON_KB_ID")
    if not kb_id:
        raise ValueError("PYLON_KB_ID not configured in .env")
    return kb_id


def _get_api_key() -> str:
    """Get Pylon API key from environment."""
    api_key = os.getenv("PYLON_API_KEY")
    if not api_key:
        raise ValueError("PYLON_API_KEY not configured in .env")
    return api_key


# =============================================================================
# Cache & API Helpers
# =============================================================================

_articles_cache: Optional[List[Dict[str, Any]]] = None
_collections_cache: Optional[Dict[str, str]] = None


def _collection_names_match(requested: str, available: str) -> bool:
    """Compare collection names without depending on parenthetical word order."""
    requested = " ".join(requested.strip().casefold().split())
    available = " ".join(available.strip().casefold().split())
    if requested == available:
        return True

    requested_match = re.fullmatch(r"(.*?)\s*\(([^()]*)\)", requested)
    available_match = re.fullmatch(r"(.*?)\s*\(([^()]*)\)", available)
    if not requested_match or not available_match:
        return False

    requested_prefix, requested_words = requested_match.groups()
    available_prefix, available_words = available_match.groups()
    return requested_prefix.strip() == available_prefix.strip() and set(
        requested_words.split()
    ) == set(available_words.split())


def _article_matches_id(article: Dict[str, Any], article_id: str) -> bool:
    """Match an article against any supported identifier form."""
    requested_id = str(article_id).strip().casefold()
    identifier = str(article.get("identifier", "")).strip()
    slug = str(article.get("slug", "")).strip()
    identifiers = {
        str(article.get("id", "")).strip(),
        identifier,
        slug,
        f"{identifier}-{slug}" if identifier and slug else "",
    }
    return requested_id in {value.casefold() for value in identifiers if value}


def _article_suggestions(articles: List[Dict[str, Any]], article_id: str) -> str:
    """Return close article title and UUID suggestions for an unknown ID."""
    requested_id = str(article_id).strip().casefold()
    ranked_articles = sorted(
        articles,
        key=lambda article: max(
            SequenceMatcher(
                None,
                requested_id,
                str(article.get(field, "")).strip().casefold(),
            ).ratio()
            for field in ("id", "identifier", "slug", "title")
        ),
        reverse=True,
    )[:3]
    return "; ".join(
        f"{article.get('title', 'Untitled')} (ID: {article.get('id')})"
        for article in ranked_articles
    )


def _get_headers() -> Dict[str, str]:
    """Get API headers with authentication."""
    return {"Authorization": f"Bearer {_get_api_key()}", "Accept": "application/json"}


def _raise_for_status(response: requests.Response, url: str) -> None:
    """Raise an operator-facing error for invalid Pylon credentials."""
    status_code = response.status_code
    if status_code in (401, 403):
        raise PylonUnavailableError(
            f"Pylon API returned HTTP {status_code} for {url}; "
            "check or rotate PYLON_API_KEY configuration or credentials."
        )
    try:
        response.raise_for_status()
    except requests.exceptions.RequestException as error:
        response_status = getattr(error.response, "status_code", None)
        if response_status in (401, 403):
            raise PylonUnavailableError(
                f"Pylon API returned HTTP {response_status} for {url}; "
                "check or rotate PYLON_API_KEY configuration or credentials."
            ) from error
        raise


def _fetch_collections() -> Dict[str, str]:
    """Fetch collections from Pylon API and cache them.

    Returns:
        Mapping of collection names to collection IDs
    """
    global _collections_cache

    if _collections_cache is not None:
        return _collections_cache

    kb_id = _get_kb_id()
    url = f"{PYLON_API_BASE_URL}/knowledge-bases/{kb_id}/collections"
    response = requests.get(url, headers=_get_headers())
    _raise_for_status(response, url)

    collections_data = response.json().get("data", [])

    # Build mapping of collection names to IDs (only public collections)
    _collections_cache = {
        coll["title"]: coll["id"]
        for coll in collections_data
        if coll.get("visibility_config", {}).get("visibility") == "public"
    }

    return _collections_cache


def _fetch_all_articles() -> List[Dict[str, Any]]:
    """Fetch all articles from Pylon API and cache them.

    Follows pagination cursors until all pages are retrieved, with a safety
    cap of 10 pages (~1000 articles) to prevent infinite loops.
    """
    global _articles_cache

    if _articles_cache is not None:
        return _articles_cache

    kb_id = _get_kb_id()
    url = f"{PYLON_API_BASE_URL}/knowledge-bases/{kb_id}/articles"
    headers = _get_headers()

    all_articles: List[Dict[str, Any]] = []
    max_pages = 10
    pages_fetched = 0
    params: Dict[str, Any] = {}

    while pages_fetched < max_pages:
        response = requests.get(url, headers=headers, params=params)
        _raise_for_status(response, url)
        body = response.json()

        page_data = body.get("data", [])
        all_articles.extend(page_data)
        pages_fetched += 1

        # Resolve next-page cursor from common Pylon/REST pagination shapes
        next_cursor = (
            body.get("next")
            or body.get("meta", {}).get("next")
            or body.get("links", {}).get("next")
            or body.get("pagination", {}).get("cursor")
        )

        if not next_cursor:
            break

        params = {"cursor": next_cursor}

    _articles_cache = all_articles
    return _articles_cache


# =============================================================================
# LangChain Tools
# =============================================================================


@tool
def search_support_articles(query: str, collections: str = "all") -> str:
    """Search published LangChain support articles using keyword text from the user's question and optional collection filters."""
    try:
        # Fetch and cache all articles (includes content)
        articles = _fetch_all_articles()

        # Handle None or empty response
        if articles is None or not articles:
            return json.dumps(
                {
                    "query": query,
                    "collections": collections,
                    "total_matched": 0,
                    "returned": 0,
                    "articles": [],
                    "note": "No articles returned from API",
                }
            )

        # Filter to only PUBLIC visibility articles with valid titles
        published_articles = []
        for article in articles:
            if (
                article.get("is_published", False)
                and article.get("title")
                and article.get("title") != "Untitled"
                and article.get("visibility_config", {}).get("visibility") == "public"
                and article.get("identifier")
                and article.get("slug")
            ):
                # Construct support.langchain.com URL
                identifier = article.get("identifier")
                slug = article.get("slug")
                support_url = (
                    f"https://support.langchain.com/articles/{identifier}-{slug}"
                )

                published_articles.append(
                    {
                        "id": article.get("id"),
                        "article_id": article.get("id"),
                        "title": article.get("title", ""),
                        "url": support_url,
                        "collection_id": article.get("collection_id"),
                        "body": article.get("current_published_content_html", ""),
                    }
                )

        if not published_articles:
            return json.dumps(
                {
                    "query": query,
                    "collections": collections,
                    "total_matched": 0,
                    "returned": 0,
                    "articles": [],
                    "note": "Support articles could not be consulted. Answer from official documentation and emit the mandatory Support articles could not be consulted disclosure.",
                }
            )

        # Fetch collection map for naming
        collection_map = _fetch_collections()

        # Filter by collection ID if specified
        unmatched_collections = []
        if collections.lower() != "all":
            # Parse requested collection names
            requested_collections = [c.strip() for c in collections.split(",")]

            # Get collection IDs for requested collections
            collection_ids = []
            for coll_name in requested_collections:
                matched_collection = next(
                    (
                        key
                        for key in collection_map
                        if _collection_names_match(coll_name, key)
                    ),
                    None,
                )
                if matched_collection is None:
                    normalized_name = " ".join(coll_name.casefold().split())
                    matched_collection = max(
                        collection_map,
                        key=lambda name: SequenceMatcher(
                            None,
                            normalized_name,
                            " ".join(name.casefold().split()),
                        ).ratio(),
                        default=None,
                    )
                    if (
                        matched_collection is None
                        or SequenceMatcher(
                            None,
                            normalized_name,
                            " ".join(matched_collection.casefold().split()),
                        ).ratio()
                        < 0.85
                    ):
                        unmatched_collections.append(coll_name)
                        continue
                collection_ids.append(collection_map[matched_collection])

            if not collection_ids:
                return json.dumps(
                    {
                        "error": f"Collection '{unmatched_collections[0]}' not found. Available collections: {', '.join(collection_map.keys())}"
                    },
                    indent=2,
                )

            # Filter articles by collection_id
            filtered_articles = [
                article
                for article in published_articles
                if article.get("collection_id") in collection_ids
            ]

            published_articles = filtered_articles

        # Update collection names based on collection_id (for all articles)
        collection_id_to_name = {v: k for k, v in collection_map.items()}
        for article in published_articles:
            coll_id = article.get("collection_id")
            article["collection"] = collection_id_to_name.get(coll_id, "Unknown")

        query_keywords = set(re.findall(r"\b[\w'-]+\b", query.casefold()))
        ranked_articles = []
        for article in published_articles:
            title_keywords = set(
                re.findall(r"\b[\w'-]+\b", article["title"].casefold())
            )
            body_text = html.unescape(re.sub(r"<[^>]+>", " ", article["body"] or ""))
            body_keywords = set(re.findall(r"\b[\w'-]+\b", body_text.casefold()))
            title_matches = query_keywords & title_keywords
            body_matches = query_keywords & body_keywords
            score = len(title_matches) * 3 + len(body_matches)
            if score == 0:
                continue

            matching_terms = title_matches or body_matches
            match_positions = [
                body_text.casefold().find(term) for term in matching_terms
            ]
            match_positions = [
                position for position in match_positions if position >= 0
            ]
            snippet_start = max(0, min(match_positions) - 100) if match_positions else 0
            snippet = body_text[snippet_start : snippet_start + 300].strip()
            ranked_articles.append(
                {
                    "score": score,
                    "article": {
                        "id": article["id"],
                        "article_id": article["id"],
                        "title": article["title"],
                        "url": article["url"],
                        "collection": article["collection"],
                        "snippet": snippet,
                    },
                }
            )

        ranked_articles.sort(key=lambda item: item["score"], reverse=True)
        if not ranked_articles:
            result = {
                "query": query,
                "collections": collections,
                "total_matched": 0,
                "returned": 0,
                "articles": [],
                "note": "Support articles could not be consulted. Answer from official documentation and emit the mandatory Support articles could not be consulted disclosure.",
            }
            if collections.lower() != "all":
                result["unmatched_collections"] = unmatched_collections
            return json.dumps(result)

        articles_to_return = [item["article"] for item in ranked_articles[:10]]
        result = {
            "query": query,
            "collections": collections,
            "total_matched": len(ranked_articles),
            "returned": len(articles_to_return),
            "articles": articles_to_return,
        }
        if collections.lower() != "all":
            result["unmatched_collections"] = unmatched_collections

        return json.dumps(result)

    except PylonUnavailableError:
        raise
    except ValueError as e:
        raise PylonUnavailableError(str(e)) from e
    except requests.exceptions.RequestException as e:
        raise PylonUnavailableError(str(e)) from e
    except Exception as e:
        raise PylonUnavailableError(str(e)) from e


@tool
def get_support_article_content(article_id: str) -> str:
    """Fetch the full HTML content of a specific Pylon support article.

    Uses cached articles from search_support_articles to avoid redundant API calls.
    This only accepts article IDs returned by search_support_articles; do not pass
    docs.langchain.com URLs or paths.

    Args:
        article_id: The article ID from search_support_articles

    Returns:
        Article content with only: id, title, url, collection, content
    """
    try:
        # Use cached articles (already fetched by search_support_articles)
        articles = _fetch_all_articles()

        # Handle None or empty response
        if articles is None or not articles:
            return "No articles available in the knowledge base."

        # Build reverse mapping: collection_id -> collection_name
        collection_map = _fetch_collections()
        collection_id_to_name = {v: k for k, v in collection_map.items()}

        # Find the article by ID
        for article in articles:
            if _article_matches_id(article, article_id):
                title = article.get("title", "Untitled")
                # Look up collection name by collection_id; fall back to default
                coll_id = article.get("collection_id")
                collection = collection_id_to_name.get(
                    coll_id, "Customer Support Knowledge Base"
                )

                # Construct support.langchain.com URL
                identifier = article.get("identifier", "")
                slug = article.get("slug", "")
                if identifier and slug:
                    support_url = (
                        f"https://support.langchain.com/articles/{identifier}-{slug}"
                    )
                else:
                    support_url = "URL not available"

                # Only return id, title, url, collection, content
                return f"""ID: {article.get("id")}
Title: {title}
URL: {support_url}
Collection: {collection}

Content:
{article.get("current_published_content_html", "No content available")[:5000]}"""

        suggestions = _article_suggestions(articles, article_id)
        return (
            f"Article ID {article_id} not found in knowledge base. "
            f"Closest matches: {suggestions}"
        )

    except PylonUnavailableError:
        raise
    except ValueError as e:
        raise PylonUnavailableError(str(e)) from e
    except requests.exceptions.RequestException as e:
        raise PylonUnavailableError(str(e)) from e
    except Exception as e:
        raise PylonUnavailableError(str(e)) from e


# Backwards-compatible Python import alias. The tool name exposed to the model is
# get_support_article_content, which avoids confusion with official docs pages.
get_article_content = get_support_article_content
