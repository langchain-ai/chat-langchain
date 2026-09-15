# Pylon Knowledge Base Tools
# Tools:
#   - search_support_articles
#   - get_support_article_content
import json
import logging
import os
import re
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
def search_support_articles(
    query: str, collections: str = "all", limit: int = 10
) -> str:
    """Search published LangChain support articles by query with optional filters."""
    try:
        # Fetch and cache all articles (includes content)
        articles = _fetch_all_articles()

        # Handle None or empty response
        if articles is None or not articles:
            return json.dumps(
                {
                    "collections": collections,
                    "query": query,
                    "total": 0,
                    "articles": [],
                    "note": "No support articles matched the query",
                },
                indent=2,
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
                        "title": article.get("title", ""),
                        "url": support_url,
                        "content": article.get("current_published_content_html", ""),
                        "collection_id": article.get(
                            "collection_id"
                        ),  # Keep for filtering, will be set later
                    }
                )

        # Fetch collection map for naming
        collection_map = _fetch_collections()

        # Filter by collection ID if specified
        if collections.lower() != "all":
            # Parse requested collection names
            requested_collections = [c.strip() for c in collections.split(",")]

            # Get collection IDs for requested collections
            collection_ids = []
            for coll_name in requested_collections:
                if coll_name in collection_map:
                    collection_ids.append(collection_map[coll_name])
                else:
                    # Try case-insensitive match
                    matched = False
                    for key in collection_map.keys():
                        if key.lower() == coll_name.lower():
                            collection_ids.append(collection_map[key])
                            matched = True
                            break
                    if not matched:
                        return json.dumps(
                            {
                                "error": (
                                    f"Collection '{coll_name}' not found. Available "
                                    f"collections: {', '.join(collection_map.keys())}"
                                ),
                                "query": query,
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

        query_terms = {
            term
            for term in re.findall(r"[a-z0-9]+", query.lower())
            if len(term) >= 3
            and term not in {
                "the",
                "and",
                "for",
                "with",
                "from",
                "how",
                "what",
                "when",
                "where",
                "why",
                "are",
                "can",
                "does",
            }
        }
        query_phrase = " ".join(re.findall(r"[a-z0-9]+", query.lower()))
        scored_articles = []
        for article in published_articles:
            title = article.get("title", "").lower()
            content = article.get("content", "").lower()
            score = (3 * sum(term in title for term in query_terms)) + sum(
                term in content for term in query_terms
            )
            if query_phrase and query_phrase in title:
                score += 2
            if score:
                article["score"] = score
                scored_articles.append(article)

        scored_articles.sort(key=lambda article: article["score"], reverse=True)
        published_articles = scored_articles[: max(limit, 0)]

        if not published_articles:
            return json.dumps(
                {
                    "collections": collections,
                    "query": query,
                    "total": 0,
                    "articles": [],
                    "note": "No support articles matched the query",
                },
                indent=2,
            )

        # Clean up collection_id from output (internal field)
        for article in published_articles:
            article.pop("collection_id", None)
            article.pop("content", None)

        # Return structured JSON format
        result = {
            "collections": collections,
            "query": query,
            "total": len(published_articles),
            "articles": published_articles,
            "note": "All articles listed are public and have content. Use IDs to fetch full content.",
        }

        return json.dumps(result, indent=2)

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
            if article.get("id") == article_id:
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

        return f"Article ID {article_id} not found in knowledge base."

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
