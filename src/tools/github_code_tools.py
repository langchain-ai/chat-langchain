"""Read-only source lookup tools for the public LangChain repositories."""

import base64
import json
import os
import re
from urllib.parse import quote

import requests
from langchain.tools import tool

GITHUB_API_BASE_URL = "https://api.github.com"
MAX_TOOL_OUTPUT_CHARS = 20_000
MAX_SEARCH_RESULTS = 10
MAX_SOURCE_CHARS = 16_000
_REPOSITORY_PATTERN = re.compile(r"^langchain-ai/[A-Za-z0-9_.-]+$")


def _validate_repository(repo: str) -> str | None:
    if not _REPOSITORY_PATTERN.fullmatch(repo):
        return (
            "Error: repo must be a public repository in the langchain-ai organization."
        )
    return None


def _github_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json, application/vnd.github.text-match+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _bounded_output(value: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    if len(value) <= limit:
        return value
    return f"{value[: limit - 40]}\n...[output truncated]"


def _request_error(response: requests.Response) -> str:
    if response.status_code == 403:
        return "Error: GitHub denied the request or the API rate limit was reached."
    if response.status_code == 404:
        return "Error: GitHub resource not found."
    return f"Error: GitHub request failed with status {response.status_code}."


@tool
def search_langchain_source(query: str, repo: str = "langchain-ai/langgraph") -> str:
    """Search source code in a public langchain-ai repository."""
    validation_error = _validate_repository(repo)
    if validation_error:
        return validation_error

    try:
        response = requests.get(
            f"{GITHUB_API_BASE_URL}/search/code",
            headers=_github_headers(),
            params={
                "q": f"{query} org:langchain-ai repo:{repo}",
                "per_page": MAX_SEARCH_RESULTS,
            },
            timeout=15,
        )
    except requests.RequestException:
        return "Error: unable to reach GitHub."

    if not response.ok:
        return _request_error(response)

    try:
        items = response.json().get("items", [])
    except (AttributeError, TypeError, ValueError):
        return "Error: GitHub returned an invalid response."

    if not items:
        return "No matching source files found."

    results: list[str] = []
    for item in items:
        fragments = [
            match.get("fragment", "") for match in item.get("text_matches", [])
        ]
        result = {
            "path": item.get("path", ""),
            "html_url": item.get("html_url", ""),
            "matches": fragments,
        }
        results.append(json.dumps(result))
    return _bounded_output("\n".join(results))


@tool
def read_langchain_source(repo: str, path: str, ref: str = "main") -> str:
    """Read a bounded slice of a source file in a public langchain-ai repository."""
    validation_error = _validate_repository(repo)
    if validation_error:
        return validation_error
    if not path or path.startswith("/") or ".." in path.split("/"):
        return "Error: path must be a relative repository file path."

    try:
        response = requests.get(
            f"{GITHUB_API_BASE_URL}/repos/{repo}/contents/{quote(path, safe='/')}",
            headers=_github_headers(),
            params={"ref": ref},
            timeout=15,
        )
    except requests.RequestException:
        return "Error: unable to reach GitHub."

    if not response.ok:
        return _request_error(response)

    try:
        payload = response.json()
        if payload.get("type") != "file":
            return "Error: GitHub path is not a file."
        content = base64.b64decode(payload["content"]).decode("utf-8")
    except (AttributeError, KeyError, TypeError, ValueError, UnicodeDecodeError):
        return "Error: GitHub returned an unreadable file response."

    return _bounded_output(
        f"path: {path}\nhtml_url: {payload.get('html_url', '')}\n\n{content}",
        MAX_SOURCE_CHARS,
    )
