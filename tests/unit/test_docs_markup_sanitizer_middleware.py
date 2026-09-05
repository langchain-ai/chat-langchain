"""Tests for removing Mintlify markup from documentation answers."""

from langchain_core.messages import AIMessage

from src.middleware.docs_markup_sanitizer_middleware import (
    DocsMarkupSanitizerMiddleware,
)


def sanitize(content: str) -> str:
    middleware = DocsMarkupSanitizerMiddleware()
    result = middleware.after_model({"messages": [AIMessage(content=content)]}, None)
    return result["messages"][0].content if result else content


def test_normalizes_fence_theme_attributes():
    assert sanitize('```python theme={"theme":{...}}\nprint("hi")\n```\n') == (
        '```python\nprint("hi")\n```\n'
    )


def test_normalizes_fence_title_and_theme_attributes():
    assert sanitize("```ts identity.ts theme={...}\nconst value = 1\n```\n") == (
        "```ts\nconst value = 1\n```\n"
    )


def test_unwraps_note_component():
    assert sanitize("<Note>text</Note>\n") == "**Note:** text\n"


def test_clean_answer_is_byte_identical():
    answer = "**Answer**\n\nUse `StateGraph`.\n"
    assert sanitize(answer) == answer
