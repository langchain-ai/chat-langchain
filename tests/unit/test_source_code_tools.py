"""Tests for allowlisted LangChain source lookup tools."""

from __future__ import annotations

import httpx
import pytest
from pydantic import ValidationError

from src.tools import source_code_tools


class FakeClient:
    def __init__(self, responses: list[httpx.Response]):
        self.responses = iter(responses)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def get(self, *args, **kwargs):
        return next(self.responses)


def response(status_code: int, content: str | bytes = "", **json_data):
    if json_data:
        import json

        content = json.dumps(json_data)
    return httpx.Response(
        status_code,
        content=content.encode() if isinstance(content, str) else content,
        request=httpx.Request("GET", "https://api.github.com"),
    )


def test_search_rejects_non_allowlisted_repo():
    with pytest.raises(ValidationError):
        source_code_tools.search_langchain_source.invoke(
            {"query": "stream", "repo": "other"}
        )


@pytest.mark.parametrize(
    "url,host",
    [
        ("https://evil.example/repos/langchain-ai/langgraph", "api.github.com"),
        ("https://api.github.com.evil.example/repos/langchain-ai/langgraph", "api.github.com"),
        ("http://api.github.com/repos/langchain-ai/langgraph", "api.github.com"),
    ],
)
def test_url_allowlist_rejects_non_github_hosts(url, host):
    with pytest.raises(ValueError, match="allowlisted"):
        source_code_tools._validate_url(url, host)


def test_read_rejects_other_owner_and_path_traversal():
    with pytest.raises(ValueError, match="allowlisted"):
        source_code_tools._validate_url(
            "https://api.github.com/repos/other/langgraph", "api.github.com"
        )
    for path in ("../secret.py", "src/../../secret.py", "/etc/passwd", "src\\secret.py"):
        with pytest.raises(ValueError, match="Invalid source path"):
            source_code_tools._validate_path(path)


def test_read_slices_lines_and_truncates_large_file(monkeypatch):
    content = "\n".join(f"line {number}" for number in range(1, 30))
    monkeypatch.setattr(source_code_tools, "MAX_SOURCE_BYTES", 40)
    client = FakeClient(
        [
            response(200, default_branch="main"),
            response(200, content),
        ]
    )
    monkeypatch.setattr(source_code_tools.httpx, "Client", lambda **kwargs: client)

    result = source_code_tools.read_langchain_source.invoke(
        {"repo": "langgraph", "path": "libs/langgraph/graph/state.py", "start_line": 2, "end_line": 4}
    )

    assert "#L2-L4" in result
    assert "2: line 2" in result
    assert "4: line 4" in result
    assert "1: line 1" not in result
    assert "output truncated at byte limit" in result
