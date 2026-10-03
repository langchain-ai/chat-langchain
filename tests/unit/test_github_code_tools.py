"""Tests for read-only GitHub source lookup tools."""

import base64
from unittest.mock import MagicMock, patch

from src.tools.github_code_tools import (
    MAX_SOURCE_CHARS,
    read_langchain_source,
    search_langchain_source,
)


def _response(payload, status_code=200):
    response = MagicMock()
    response.ok = status_code < 400
    response.status_code = status_code
    response.json.return_value = payload
    return response


def test_search_rejects_repositories_outside_langchain_org():
    result = search_langchain_source.invoke(
        {"query": "stream", "repo": "other-org/project"}
    )

    assert "langchain-ai organization" in result


@patch("src.tools.github_code_tools.requests.get")
def test_search_returns_paths_urls_and_matches(mock_get):
    mock_get.return_value = _response(
        {
            "items": [
                {
                    "path": "libs/langgraph/langgraph/pregel/main.py",
                    "html_url": "https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/pregel/main.py",
                    "text_matches": [{"fragment": "stream_mode"}],
                }
            ]
        }
    )

    result = search_langchain_source.invoke(
        {"query": "stream_mode", "repo": "langchain-ai/langgraph"}
    )

    assert "libs/langgraph/langgraph/pregel/main.py" in result
    assert "https://github.com/langchain-ai/langgraph/blob/main" in result
    assert "stream_mode" in result
    assert "repo:langchain-ai/langgraph" in mock_get.call_args.kwargs["params"]["q"]


@patch("src.tools.github_code_tools.requests.get")
def test_read_returns_bounded_file_content_and_url(mock_get):
    mock_get.return_value = _response(
        {
            "type": "file",
            "html_url": "https://github.com/langchain-ai/langgraph/blob/main/file.py",
            "content": base64.b64encode(b"x" * MAX_SOURCE_CHARS).decode(),
        }
    )

    result = read_langchain_source.invoke(
        {"repo": "langchain-ai/langgraph", "path": "file.py"}
    )

    assert len(result) <= MAX_SOURCE_CHARS
    assert "https://github.com/langchain-ai/langgraph/blob/main/file.py" in result
    assert "[output truncated]" in result


@patch("src.tools.github_code_tools.requests.get")
def test_github_rate_limit_returns_short_error(mock_get):
    mock_get.return_value = _response({}, status_code=403)

    result = search_langchain_source.invoke({"query": "stream"})

    assert "rate limit" in result
    assert len(result) < 200
