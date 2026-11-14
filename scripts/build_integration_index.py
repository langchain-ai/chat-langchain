"""Rebuild src/tools/integration_pages.json from the langchain-ai/docs repo.

Run when integration pages are added/renamed: python scripts/build_integration_index.py
"""

import concurrent.futures as cf
import json
import re
import urllib.request
from pathlib import Path

REPO = "langchain-ai/docs"
OUT = Path(__file__).resolve().parent.parent / "src/tools/integration_pages.json"
LEAF = re.compile(r"src/oss/(python|javascript)/integrations/[^/]+/[^/]+\.mdx$")
CLASS = re.compile(
    r"\b(Chat[A-Z][A-Za-z0-9]+|[A-Z][A-Za-z0-9]+(?:Loader|Embeddings|VectorStore|Retriever))\b"
)


def _get(url: str) -> str:
    return urllib.request.urlopen(url, timeout=30).read().decode()


def main() -> None:
    """Fetch every integration leaf page and write path/title/name/classes."""
    tree = json.loads(
        _get(f"https://api.github.com/repos/{REPO}/git/trees/main?recursive=1")
    )
    paths = [
        t["path"]
        for t in tree["tree"]
        if LEAF.match(t["path"]) and not t["path"].endswith("/index.mdx")
    ]

    def fetch(path: str) -> dict:
        text = _get(f"https://raw.githubusercontent.com/{REPO}/main/{path}")
        title = re.search(r'^title:\s*"?(.+?)"?\s*$', text, re.M)
        name = re.search(r'^\s+name:\s*"?(.+?)"?\s*$', text, re.M)
        return {
            "path": path[3:-4],
            "title": title[1] if title else "",
            "name": name[1] if name else "",
            "classes": sorted(set(CLASS.findall(text)))[:25],
        }

    with cf.ThreadPoolExecutor(16) as pool:
        entries = sorted(pool.map(fetch, paths), key=lambda e: e["path"])
    OUT.write_text(json.dumps(entries, indent=0))
    print(f"wrote {len(entries)} pages to {OUT}")  # noqa: T201


if __name__ == "__main__":
    main()
