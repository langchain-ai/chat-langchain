import base64
from unittest.mock import Mock, patch

from src.tools.source_code_tools import read_source_file, search_source_code


def _response(payload, status_code=200):
    response = Mock()
    response.ok = status_code < 400
    response.status_code = status_code
    response.json.return_value = payload
    return response


def test_search_source_code_formats_results():
    response = _response(
        {
            "items": [
                {
                    "path": "libs/langgraph/langgraph/graph/state.py",
                    "html_url": "https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/graph/state.py",
                    "text_matches": [{"fragment": "  return stream_mode  "}],
                }
            ]
        }
    )

    with patch("src.tools.source_code_tools.requests.get", return_value=response) as get:
        result = search_source_code.invoke({"query": "subgraph streaming", "repo": "langgraph"})

    assert "libs/langgraph/langgraph/graph/state.py" in result
    assert "Snippet: return stream_mode" in result
    assert "https://github.com/langchain-ai/langgraph/blob/main" in result
    assert get.call_args.kwargs["params"]["q"].endswith("repo:langchain-ai/langgraph")


def test_read_source_file_slices_and_numbers_lines():
    content = "one\ntwo\nthree\nfour\n"
    responses = [
        _response({"default_branch": "main"}),
        _response(
            {
                "type": "file",
                "content": base64.b64encode(content.encode()).decode(),
            }
        ),
    ]

    with patch("src.tools.source_code_tools.requests.get", side_effect=responses):
        result = read_source_file.invoke(
            {"repo": "langgraph", "path": "libs/core.py", "start_line": 2, "max_lines": 2}
        )

    assert result.startswith("2: two\n3: three")
    assert "https://github.com/langchain-ai/langgraph/blob/main/libs/core.py" in result


def test_source_tools_reject_repository_and_path_traversal():
    assert "Invalid repository" in search_source_code.invoke({"query": "x", "repo": "other"})
    assert "Path traversal" in read_source_file.invoke(
        {"repo": "langgraph", "path": "libs/../secret.py"}
    )
    assert "Invalid path" in read_source_file.invoke(
        {"repo": "langgraph", "path": "/etc/passwd"}
    )


def test_source_tool_truncates_output():
    response = _response(
        {
            "items": [
                {
                    "path": f"file-{index}.py",
                    "html_url": "https://github.com/langchain-ai/langchain",
                    "text_matches": [{"fragment": "x" * 300}],
                }
                for index in range(10)
            ]
        }
    )
    for item in response.json.return_value["items"]:
        item["path"] = "x" * 2_000

    with patch("src.tools.source_code_tools.requests.get", return_value=response):
        result = search_source_code.invoke({"query": "x", "repo": "langchain"})

    assert result.endswith("[Output truncated.]")
    assert len(result) <= 12_000 + len("\n[Output truncated.]")


def test_source_tools_return_clear_http_errors():
    with patch(
        "src.tools.source_code_tools.requests.get",
        return_value=_response({}, status_code=429),
    ):
        assert "rate limit" in search_source_code.invoke({"query": "x", "repo": "langchain"})

    responses = [
        _response({"default_branch": "main"}, status_code=200),
        _response({}, status_code=404),
    ]
    with patch("src.tools.source_code_tools.requests.get", side_effect=responses):
        assert "not found" in read_source_file.invoke(
            {"repo": "langchain", "path": "missing.py"}
        )

    with patch(
        "src.tools.source_code_tools.requests.get",
        return_value=_response({}, status_code=500),
    ):
        assert "HTTP 500" in search_source_code.invoke({"query": "x", "repo": "langchain"})
