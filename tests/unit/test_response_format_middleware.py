"""Tests for generated code block formatting."""

from langchain_core.messages import AIMessage

from src.middleware.response_format_middleware import (
    ResponseFormatMiddleware,
    normalize_python_fences,
)


def test_normalize_python_fence_converts_leading_javascript_comments():
    text = "```python\n// explain this line\nvalue = 1\n```"

    assert normalize_python_fences(text) == "```python\n# explain this line\nvalue = 1\n```"


def test_normalize_python_fence_preserves_strings_and_urls():
    text = (
        "```python\n"
        "url = \"https://docs.example.com/a//b\"\n"
        "message = \"// keep this text\"\n"
        "description = \"\"\"\n"
        "// keep this string content\n"
        "\"\"\"\n"
        "```"
    )

    assert normalize_python_fences(text) == text


def test_normalize_python_fence_does_not_change_typescript():
    text = "```typescript\n// keep this comment\nconst value = 1\n```"

    assert normalize_python_fences(text) == text


def test_normalize_python_fence_warns_about_javascript_indicators(caplog):
    text = "```python\nconst value = 1\nvalue => value\n```"

    normalize_python_fences(text)

    assert "JavaScript indicators" in caplog.text
    assert text == normalize_python_fences(text)


def test_response_format_middleware_updates_latest_ai_message():
    message = AIMessage(content="```py\n// comment\nvalue = 1\n```")
    state = {"messages": [message]}

    update = ResponseFormatMiddleware().after_model(state, runtime=None)

    assert update is not None
    assert update["messages"][0].content == "```py\n# comment\nvalue = 1\n```"
