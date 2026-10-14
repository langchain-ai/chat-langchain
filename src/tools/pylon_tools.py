# Pylon Knowledge Base Tools
# Tools:
#   - search_support_articles
#   - get_support_article_content
import json
import logging
import os
import re
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv()

logger = logging.getLogger(__name__)

# Pylon API configuration
PYLON_API_BASE_URL = "https://api.usepylon.com"
ARTICLE_CONTENT_MAX_CHARS = 12000


class _ArticleHTMLParser(HTMLParser):
    """Render article HTML as readable text with intact links and code fences."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: List[tuple[str, str]] = []
        self.capture_tag = ""
        self.buffer: List[str] = []
        self.href = ""
        self.hidden_tag = ""

    def handle_starttag(self, tag: str, attrs: List[tuple[str, str | None]]) -> None:
        """Capture links and code, separating block elements from surrounding text."""
        if self.hidden_tag:
            return
        if tag in {"script", "style"}:
            self.hidden_tag = tag
        elif not self.capture_tag:
            if tag in {"a", "pre", "code"}:
                self.capture_tag = tag
                self.href = dict(attrs).get("href") or ""
            elif tag == "br":
                self.parts.append(("\n", ""))
            elif tag in {
                "p",
                "div",
                "li",
                "ul",
                "ol",
                "blockquote",
                "tr",
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "h6",
            }:
                self._break()
        elif tag == "br":
            self.buffer.append("\n")

    def handle_endtag(self, tag: str) -> None:
        """Finish captured elements and separate adjacent blocks."""
        if self.hidden_tag:
            if tag == self.hidden_tag:
                self.hidden_tag = ""
            return
        if self.capture_tag:
            if tag == self.capture_tag:
                self.finish_capture()
        elif tag in {
            "p",
            "div",
            "li",
            "ul",
            "ol",
            "blockquote",
            "tr",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
        }:
            self._break()

    def handle_data(self, data: str) -> None:
        """Unescape text via HTMLParser while preserving code whitespace."""
        if self.hidden_tag:
            return
        if self.capture_tag in {"pre", "code"}:
            self.buffer.append(data)
        else:
            text = re.sub(r"\s+", " ", data)
            if self.capture_tag:
                self.buffer.append(text)
            elif text.strip() or (self.parts and not self.parts[-1][0].endswith("\n")):
                self.parts.append((text, ""))

    def _break(self) -> None:
        """Separate blocks without accumulating blank lines."""
        if self.parts and not self.parts[-1][0].endswith("\n\n"):
            self.parts.append(("\n\n", ""))

    def finish_capture(self) -> None:
        """Render a captured link or code block as one protected segment."""
        if not self.capture_tag:
            return
        text = "".join(self.buffer)
        if self.capture_tag == "a" and self.href:
            label = (
                text.strip()
                .replace("\\", "\\\\")
                .replace("[", "\\[")
                .replace("]", "\\]")
            )
            target = self.href.replace("(", "%28").replace(")", "%29")
            self.parts.append((f"[{label}]({target})", "link"))
        elif self.capture_tag in {"pre", "code"}:
            longest_run = max((len(run) for run in re.findall(r"`+", text)), default=0)
            fence = "`" * max(3, longest_run + 1)
            code = text.strip("\n")
            self.parts.append((f"\n\n{fence}\n{code}\n{fence}\n\n", fence))
        else:
            self.parts.append((text, ""))
        self.capture_tag = ""
        self.buffer = []


def _format_article_content(html: str) -> str:
    """Convert full HTML before limiting content and marking any truncation."""
    parser = _ArticleHTMLParser()
    parser.feed(html)
    parser.close()
    parser.finish_capture()
    rendered = "".join(text for text, _ in parser.parts)
    content = rendered.strip()
    total = len(content)
    cap = ARTICLE_CONTENT_MAX_CHARS
    if total <= cap:
        return content

    shown = content[:cap]
    offset = -(len(rendered) - len(rendered.lstrip()))
    for text, kind in parser.parts:
        end = offset + len(text)
        if kind and offset < cap < end:
            if kind == "link":
                shown = content[: max(0, offset)].rstrip()
            elif cap < end - 2:
                closing = f"\n{kind}"
                if cap - offset >= 2 * len(kind) + 4:
                    shown = content[: cap - len(closing)] + closing
                else:
                    shown = content[: max(0, offset)].rstrip()
            break
        offset = end
    return f"{shown}\n\n[Article truncated: showing {len(shown)} of {total} characters]"


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
    response.raise_for_status()

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
        response.raise_for_status()
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
def search_support_articles(collections: str = "all") -> str:
    """Get LangChain support article titles from Pylon KB, filtered by collection(s).

    Returns article titles in structured JSON format so the LLM can decide which ones to fetch.

    Args:
        collections: Comma-separated list of collection names to filter by.
                    Available collections:
                    - "General" - General administration and management topics
                    - "OSS (LangChain and LangGraph)" - Open source libraries for LangChain and LangGraph
                    - "LangSmith Observability" - Tracing, stats, and observability of agents
                    - "LangSmith Evaluation" - Datasets, evaluations, and prompts
                    - "LangSmith Deployment" - Graph runtime and deployments (formerly LangGraph Platform)
                    - "SDKs and APIs" - All things across SDKs and APIs
                    - "LangSmith Studio" - Visualizing and debugging agents (formerly LangGraph Studio)
                    - "Self Hosted" - Self-hosted LangSmith including deployments
                    - "Troubleshooting" - Broad domain issue triage and resolution
                    - "Security" - Code scans, key management, and security topics

                    Use "all" to search all collections (default)
                    Example: "LangSmith Deployment,LangSmith Observability" to get articles about both

    Returns:
        JSON string with structure: {"collections": "...", "total": N, "articles": [...]}
    """
    try:
        # Fetch and cache all articles (includes content)
        articles = _fetch_all_articles()

        # Handle None or empty response
        if articles is None or not articles:
            return json.dumps(
                {
                    "collections": collections,
                    "total": 0,
                    "articles": [],
                    "note": "No articles returned from API",
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
                        "collection_id": article.get(
                            "collection_id"
                        ),  # Keep for filtering, will be set later
                    }
                )

        if not published_articles:
            return "No published articles available in the knowledge base."

        # Fetch collection map for naming
        try:
            collection_map = _fetch_collections()
        except Exception as e:
            return json.dumps(
                {"error": f"Failed to fetch collections: {str(e)}"}, indent=2
            )

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
                                "error": f"Collection '{coll_name}' not found. Available collections: {', '.join(collection_map.keys())}"
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

        if not published_articles:
            return json.dumps(
                {
                    "collections": collections,
                    "total": 0,
                    "articles": [],
                    "note": "No articles found",
                },
                indent=2,
            )

        # Clean up collection_id from output (internal field)
        for article in published_articles:
            article.pop("collection_id", None)

        # Return structured JSON format
        result = {
            "collections": collections,
            "total": len(published_articles),
            "articles": published_articles,
            "note": "All articles listed are public and have content. Use IDs to fetch full content.",
        }

        return json.dumps(result, indent=2)

    except ValueError as e:
        # API key not configured
        return json.dumps({"error": str(e)}, indent=2)
    except requests.exceptions.RequestException as e:
        # Network/API error
        return json.dumps({"error": str(e)}, indent=2)
    except Exception as e:
        # Catch-all for unexpected errors
        return json.dumps({"error": f"Unexpected error: {str(e)}"}, indent=2)


@tool
def get_support_article_content(article_id: str) -> str:
    """Fetch readable Pylon support content by search_support_articles ID, with marked truncation."""
    try:
        # Use cached articles (already fetched by search_support_articles)
        articles = _fetch_all_articles()

        # Handle None or empty response
        if articles is None or not articles:
            return "Error: No articles available from API. Check PYLON_API_KEY configuration."

        # Build reverse mapping: collection_id -> collection_name
        try:
            collection_map = _fetch_collections()
            collection_id_to_name = {v: k for k, v in collection_map.items()}
        except Exception:
            collection_id_to_name = {}

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
{_format_article_content(article.get("current_published_content_html", "No content available"))}"""

        return f"Article ID {article_id} not found in knowledge base."

    except ValueError as e:
        # API key not configured
        return f"Error: {str(e)}"
    except requests.exceptions.RequestException as e:
        # Network/API error
        return f"Error fetching article: {str(e)}"
    except Exception as e:
        # Catch-all for unexpected errors
        return f"Unexpected error: {str(e)}"


# Backwards-compatible Python import alias. The tool name exposed to the model is
# get_support_article_content, which avoids confusion with official docs pages.
get_article_content = get_support_article_content
