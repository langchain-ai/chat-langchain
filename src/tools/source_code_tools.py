"""Read-only source lookup tools for public LangChain repositories."""

import base64
import json
import re
from urllib.parse import quote

import httpx
from langchain.tools import tool

GITHUB_API_URL = "https://api.github.com"
GITHUB_USER_AGENT = "LangChain-SourceReader/1.0"
GITHUB_TIMEOUT = 15.0
ALLOWED_REPOSITORIES = {
    "langchain": "langchain-ai/langchain",
    "langgraph": "langchain-ai/langgraph",
    "deepagents": "langchain-ai/deepagents",
}
MAX_RESULTS = 5
MAX_FILE_BYTES = 100_000
MAX_PATH_LENGTH = 200
MAX_QUERY_LENGTH = 200
MAX_FILES_TO_SCAN = 30
SOURCE_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".go",
    ".java",
    ".js",
    ".jsx",
    ".md",
    ".mdx",
    ".py",
    ".rs",
    ".ts",
    ".tsx",
    ".yaml",
    ".yml",
}


def _validate_inputs(repository: str, path: str, query: str) -> str | None:
    """Validate repository source lookup inputs."""
    if repository not in ALLOWED_REPOSITORIES:
        return "Error: repository must be one of: langchain, langgraph, deepagents."
    if len(path) > MAX_PATH_LENGTH:
        return f"Error: path must be at most {MAX_PATH_LENGTH} characters."
    if len(query) > MAX_QUERY_LENGTH:
        return f"Error: query must be at most {MAX_QUERY_LENGTH} characters."
    if path.startswith("/") or ".." in path.split("/"):
        return "Error: path must be a relative repository path without '..'."
    return None


async def _get_json(client: httpx.AsyncClient, url: str) -> dict | list:
    """Fetch a GitHub JSON response."""
    response = await client.get(url)
    response.raise_for_status()
    return response.json()


def _contents_url(repository: str, path: str, branch: str) -> str:
    """Build a GitHub contents API URL."""
    encoded_path = quote(path, safe="/")
    return (
        f"{GITHUB_API_URL}/repos/{repository}/contents/{encoded_path}"
        f"?ref={quote(branch, safe='')}"
    )


def _source_entries(tree: list[dict], path: str) -> list[dict]:
    """Select bounded source file entries from a Git tree."""
    entries = []
    for entry in tree:
        entry_path = entry.get("path", "")
        if entry.get("type") != "blob" or len(entry_path) > MAX_PATH_LENGTH:
            continue
        if path and not (entry_path == path or entry_path.startswith(f"{path}/")):
            continue
        if any(entry_path.endswith(extension) for extension in SOURCE_EXTENSIONS):
            entries.append(entry)
    return entries


def _rank_entries(entries: list[dict], query: str) -> list[dict]:
    """Rank source entries by query terms in their paths."""
    terms = [term for term in re.findall(r"[a-z0-9_]+", query.lower()) if len(term) > 1]

    def score(entry: dict) -> tuple[int, int, str]:
        entry_path = entry["path"].lower()
        path_score = sum(entry_path.count(term) for term in terms)
        return (-path_score, len(entry_path), entry_path)

    return sorted(entries, key=score)


def _snippet(content: str, query: str) -> str:
    """Extract a bounded source snippet around query matches."""
    if not query:
        return content[:12_000]
    terms = [term for term in re.findall(r"[a-z0-9_]+", query.lower()) if len(term) > 1]
    lines = content.splitlines()
    matching_lines = [
        index
        for index, line in enumerate(lines)
        if any(term in line.lower() for term in terms)
    ]
    if not matching_lines:
        return content[:12_000]
    selected: list[str] = []
    for index in matching_lines[:8]:
        selected.extend(lines[max(0, index - 3) : index + 4])
    return "\n".join(selected)[:12_000]


async def _read_file(
    client: httpx.AsyncClient,
    repository: str,
    branch: str,
    entry: dict,
    query: str,
) -> dict:
    """Read one bounded source file from GitHub."""
    path = entry["path"]
    if entry.get("size", 0) > MAX_FILE_BYTES:
        return {
            "path": path,
            "url": f"https://github.com/{repository}/blob/{branch}/{path}",
            "error": f"File exceeds the {MAX_FILE_BYTES}-byte limit.",
        }
    payload = await _get_json(client, _contents_url(repository, path, branch))
    if payload.get("encoding") != "base64":
        return {"path": path, "error": "GitHub did not return base64 file content."}
    content = base64.b64decode(payload["content"]).decode("utf-8", errors="replace")
    return {
        "path": path,
        "url": f"https://github.com/{repository}/blob/{branch}/{path}",
        "content": content if not query else _snippet(content, query),
        "matches_query": not query or _matches_query(content, query),
    }


def _matches_query(content: str, query: str) -> bool:
    """Check whether source content contains every query term."""
    terms = [term for term in re.findall(r"[a-z0-9_]+", query.lower()) if len(term) > 1]
    return all(term in content.lower() for term in terms)


@tool
async def search_langchain_source(
    repository: str,
    query: str = "",
    path: str = "",
    operation: str = "search",
    max_results: int = MAX_RESULTS,
) -> str:
    """Search or read source files in the allowed public LangChain repositories."""
    validation_error = _validate_inputs(repository, path, query)
    if validation_error:
        return validation_error
    if operation not in {"search", "read"}:
        return "Error: operation must be 'search' or 'read'."
    if operation == "read" and not path:
        return "Error: path is required when operation is 'read'."
    if operation == "search" and not query and not path:
        return "Error: provide a query or path when operation is 'search'."

    result_count = max(1, min(max_results, MAX_RESULTS))
    full_repository = ALLOWED_REPOSITORIES[repository]
    headers = {"Accept": "application/vnd.github+json", "User-Agent": GITHUB_USER_AGENT}
    try:
        async with httpx.AsyncClient(headers=headers, timeout=GITHUB_TIMEOUT) as client:
            metadata = await _get_json(
                client, f"{GITHUB_API_URL}/repos/{full_repository}"
            )
            branch = metadata["default_branch"]
            if operation == "read":
                payload = await _get_json(
                    client, _contents_url(full_repository, path, branch)
                )
                entry = {"path": path, "size": payload.get("size", 0)}
                result = await _read_file(client, full_repository, branch, entry, "")
                return json.dumps(
                    {
                        "repository": full_repository,
                        "branch": branch,
                        "results": [result],
                    }
                )

            tree = await _get_json(
                client,
                f"{GITHUB_API_URL}/repos/{full_repository}/git/trees/"
                f"{quote(branch, safe='')}?recursive=1",
            )
            entries = _source_entries(tree.get("tree", []), path)
            entries = _rank_entries(entries, query)[:MAX_FILES_TO_SCAN]
            results = []
            for entry in entries:
                result = await _read_file(client, full_repository, branch, entry, query)
                if "content" in result and query:
                    if not result["matches_query"]:
                        continue
                result.pop("matches_query", None)
                results.append(result)
                if len(results) >= result_count:
                    break
            return json.dumps(
                {"repository": full_repository, "branch": branch, "results": results}
            )
    except httpx.HTTPStatusError as exc:
        return f"Error: GitHub returned HTTP {exc.response.status_code} while reading {full_repository}."
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        return f"Error: unable to read {full_repository} from GitHub: {exc}."
