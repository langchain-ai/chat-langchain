"""Find and read LangChain integration pages that the docs MCP does not index.

The docs MCP (search + filesystem) only covers pages in the site navigation.
About 490 integration leaf pages (e.g. /oss/python/integrations/document_loaders/docling,
/oss/javascript/integrations/chat/ollama) are live on docs.langchain.com but absent from
docs search, the docs filesystem, llms.txt and the sitemap, so the agent kept landing on
category index pages and re-searching. This tool matches a class/provider name against a
bundled list of those pages (built from the langchain-ai/docs repo) and reads the page's
markdown from docs.langchain.com.
"""

import json
import logging
import re
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Literal

import httpx
from langchain.tools import tool

logger = logging.getLogger(__name__)

DOCS_BASE_URL = "https://docs.langchain.com"
INDEX_FILE = Path(__file__).with_name("integration_pages.json")
TIMEOUT = 10.0
MAX_PAGE_CHARS = 12_000
MAX_CANDIDATES = 5
MIN_READ_SCORE = 60
_CACHE_TTL_SECONDS = 3600
_cache_lock = threading.Lock()
_page_cache: dict[str, tuple[float, str]] = {}


@lru_cache(maxsize=1)
def _load_index() -> list[dict]:
    """Load the bundled integration page list."""
    with INDEX_FILE.open() as f:
        return json.load(f)


def _norm(text: str) -> str:
    """Lowercase and drop non-alphanumerics so 'Chat Ollama' == 'chat_ollama'."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


_PREFIXES = ("langchain", "chat")
_SUFFIXES = (
    "documentloader",
    "loader",
    "embeddings",
    "vectorstore",
    "retriever",
    "toolkit",
    "tool",
    "llm",
)


def _stem(q: str) -> str:
    """Strip package/class affixes: 'langchain-docling' / 'DoclingLoader' -> 'docling'."""
    for prefix in _PREFIXES:
        if q.startswith(prefix) and len(q) > len(prefix) + 2:
            q = q[len(prefix) :]
    for suffix in _SUFFIXES:
        if q.endswith(suffix) and len(q) > len(suffix) + 2:
            q = q[: -len(suffix)]
            break
    return q


def _score(entry: dict, query: str) -> int:
    """Score how well an index entry matches a class/provider name."""
    q = _norm(query)
    if not q:
        return 0
    slug = _norm(entry["path"].rsplit("/", 1)[-1])
    name = _norm(entry.get("name", ""))
    title = _norm(
        entry.get("title", "").replace("integrations", "").replace("integration", "")
    )
    classes = {_norm(c) for c in entry.get("classes", [])}
    score = 0
    if q in (name, slug, title):
        score += 100
    if q in classes:
        # Classes found in a page's code are often incidental (PyPDFLoader in
        # cloud-storage examples, WebBaseLoader in a reranker page), so a class
        # mention alone never makes a page the match; it only breaks ties.
        score += 10
    stem = _stem(q)
    if stem and stem != q and stem in (slug, name, title):
        score += 60
    if len(q) >= 4 and (q in slug or (slug and len(slug) >= 4 and slug in q)):
        score += 30
    if len(q) >= 4 and any(q in c for c in classes):
        score += 10
    if "/providers/" in entry["path"]:
        score -= 5  # prefer the component page over the provider overview
    return score


def find_candidates(query: str, language: str) -> list[dict]:
    """Return the best-matching integration pages for a name in one language."""
    prefix = f"/oss/{language}/integrations/"
    scored = [
        (_score(e, query), e) for e in _load_index() if e["path"].startswith(prefix)
    ]
    scored = [s for s in scored if s[0] >= MIN_READ_SCORE]
    scored.sort(key=lambda s: (-s[0], s[1]["path"]))
    return [e for _, e in scored[:MAX_CANDIDATES]]


def _strip_preamble(markdown: str) -> str:
    """Drop the site's 'Documentation Index' blockquote at the top of .md pages."""
    lines = markdown.splitlines()
    i = 0
    while i < len(lines) and (lines[i].startswith(">") or not lines[i].strip()):
        i += 1
    return "\n".join(lines[i:]).strip()


async def _fetch_page(path: str) -> str:
    """Fetch a page's markdown (cached per process)."""
    now = time.monotonic()
    with _cache_lock:
        hit = _page_cache.get(path)
        if hit and now - hit[0] < _CACHE_TTL_SECONDS:
            return hit[1]
    async with httpx.AsyncClient(follow_redirects=True) as client:
        response = await client.get(f"{DOCS_BASE_URL}{path}.md", timeout=TIMEOUT)
        response.raise_for_status()
        text = _strip_preamble(response.text)
    with _cache_lock:
        _page_cache[path] = (now, text)
    return text


@tool
async def read_integration_page(
    name: str, language: Literal["python", "javascript"] = "python"
) -> str:
    """Read the docs page for one specific integration: a provider, chat model, embedding model, document loader, vector store, retriever, tool, checkpointer or similar component (e.g. "DoclingLoader", "ChatOllama", "Pinecone", "PyMuPDFLoader", "Tavily").

    Use it whenever the question names a specific integration, in parallel with docs search. Docs search and the docs filesystem do not contain most individual integration pages (they only return category index pages such as /oss/python/integrations/document_loaders/index), so don't conclude an integration is undocumented until you've called this.

    Args:
        name: The class, package or provider name, e.g. "DoclingLoader", "ChatOllama", "ollama".
        language: "python" or "javascript" (the user's stack).
    """
    candidates = find_candidates(name, language)
    if not candidates:
        return (
            f"No dedicated {language} integration page matches '{name}'. "
            "Use the docs search results / integrations index pages instead."
        )
    top = candidates[0]
    others = "\n".join(
        f"- {DOCS_BASE_URL}{c['path']} ({c['title']})" for c in candidates[1:]
    )
    try:
        page = await _fetch_page(top["path"])
    except Exception as exc:  # noqa: BLE001 - tool must never raise into the agent
        logger.warning("integration page fetch failed for %s: %s", top["path"], exc)
        return (
            f"Best match: {DOCS_BASE_URL}{top['path']} ({top['title']}), but it could not be "
            f"read right now. Other matches:\n{others}"
        )
    if len(page) > MAX_PAGE_CHARS:
        page = page[:MAX_PAGE_CHARS].rstrip() + "\n\n[page truncated]"
    header = f"Page: {DOCS_BASE_URL}{top['path']}\nTitle: {top['title']}\n"
    if others:
        header += f"Other matching pages:\n{others}\n"
    return f"{header}\n{page}"
