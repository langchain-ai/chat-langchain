import base64
from unittest.mock import Mock, patch

from src.tools.source_tools import (
    MAX_FILE_CONTENT_CHARS,
    MAX_SEARCH_RESULTS,
    read_langchain_source,
    search_langchain_source,
)


def test_source_tools_reject_unallowlisted_repository_before_request():
    with patch("src.tools.source_tools.requests.get") as mock_get:
        result = search_langchain_source.invoke(
            {"query": "StateGraph", "repo": "example/project"}
        )

    assert result.startswith("Error: repository must be one of:")
    mock_get.assert_not_called()


def test_search_source_caps_results_and_includes_file_urls():
    response = Mock(status_code=200)
    response.json.return_value = {
        "items": [
            {"path": f"src/file_{index}.py", "html_url": f"https://github.com/file/{index}"}
            for index in range(MAX_SEARCH_RESULTS + 3)
        ]
    }

    with patch("src.tools.source_tools.requests.get", return_value=response) as mock_get:
        result = search_langchain_source.invoke(
            {"query": "StateGraph", "repo": "langchain-ai/langgraph"}
        )

    assert result.count("https://github.com/file/") == MAX_SEARCH_RESULTS
    mock_get.assert_called_once()


def test_read_source_caps_file_content_and_includes_file_url():
    content = "x" * (MAX_FILE_CONTENT_CHARS + 20)
    response = Mock(status_code=200)
    response.json.return_value = {
        "content": base64.b64encode(content.encode()).decode(),
        "html_url": "https://github.com/langchain-ai/langgraph/blob/main/file.py",
    }

    with patch("src.tools.source_tools.requests.get", return_value=response):
        result = read_langchain_source.invoke(
            {"repo": "langchain-ai/langgraph", "path": "file.py"}
        )

    assert "https://github.com/langchain-ai/langgraph/blob/main/file.py" in result
    assert f"[Content truncated to {MAX_FILE_CONTENT_CHARS} characters.]" in result
    assert result.count("x") == MAX_FILE_CONTENT_CHARS
