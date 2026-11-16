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
CODE_BLOCK = re.compile(
    r"```(?:python|py|javascript|js|typescript|ts)\b[^\n]*\n(.*?)```", re.S
)
PYTHON_IMPORT = re.compile(
    r"^\s*from\s+(langchain[\w.]*)\s+import\s+(\([^)]*\)|[^\n]+)", re.M
)
JS_IMPORT = re.compile(
    r"\bimport\s+(\{[^}]+\}|[A-Z]\w*)\s+from\s+['\"](@langchain/[^'\"]+|langchain[^'\"]*)['\"]",
    re.S,
)
GENERIC_CLASSES = {
    "ChatOpenAI",
    "OpenAIEmbeddings",
    "ChatPromptTemplate",
    "InMemoryVectorStore",
    "RecursiveCharacterTextSplitter",
}


def _extract_classes(text: str, path: str, title: str, name: str) -> list[str]:
    """Extract imported classes whose names identify this integration."""
    slug = path.rsplit("/", 1)[-1].removesuffix(".mdx")
    identities = {
        re.sub(r"[^a-z0-9]", "", value.lower())
        for value in (slug, title.replace("integration", ""), name)
    } - {""}
    tokens = set(re.split(r"[_-]", slug)) - {
        "search",
        "chat",
        "tool",
        "tools",
        "loader",
        "embeddings",
        "vectorstore",
    }
    classes = set()
    for block in CODE_BLOCK.findall(text):
        imports = [match[1] for match in PYTHON_IMPORT.findall(block)]
        imports.extend(match[0] for match in JS_IMPORT.findall(block))
        for imported in imports:
            for item in imported.strip("(){} \n").split(","):
                symbol = item.strip().split(" as ")[0].strip()
                if not re.fullmatch(r"[A-Z][A-Za-z0-9]+", symbol):
                    continue
                normalized = symbol.lower()
                if normalized in identities or (
                    symbol not in GENERIC_CLASSES
                    and any(len(t) >= 3 and t in normalized for t in tokens)
                ):
                    classes.add(symbol)
    return sorted(classes)


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
        classes = _extract_classes(
            text, path, title[1] if title else "", name[1] if name else ""
        )
        return {
            "path": path[3:-4],
            "title": title[1] if title else "",
            "name": name[1] if name else "",
            "classes": classes,
            "verified_classes": classes,
        }

    with cf.ThreadPoolExecutor(16) as pool:
        entries = sorted(pool.map(fetch, paths), key=lambda e: e["path"])
    OUT.write_text(json.dumps(entries, indent=0))
    print(f"wrote {len(entries)} pages to {OUT}")  # noqa: T201


if __name__ == "__main__":
    main()
