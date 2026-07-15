"""Tests for repairing non-Python syntax leaking into python code fences."""

from __future__ import annotations

import os
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

os.environ["USE_LOCAL_PROMPTS"] = "1"

from src.middleware.code_fence_validator_middleware import (
    CodeFenceValidatorMiddleware,
)


def _run(content):
    middleware = CodeFenceValidatorMiddleware()
    state = {"messages": [HumanMessage(content="hi"), AIMessage(content=content, id="a1")]}
    return middleware.after_agent(state, runtime=SimpleNamespace())


def test_rewrites_js_line_comment_in_python_fence():
    content = "```python\n# 1. Intercept the request\nx = 1  // not python\n```"
    update = _run(content)
    assert update is not None
    fixed = update["messages"][0].content
    assert "//" not in fixed
    assert "# not python" in fixed


def test_rewrites_arrow_function_in_wrap_model_call_example():
    content = (
        "Here is a middleware hook:\n\n"
        "```python\n"
        "wrap_model_call = (req, handler) => handler(req)  // pass through\n"
        "```\n"
    )
    update = _run(content)
    assert update is not None
    fixed = update["messages"][0].content
    assert "=>" not in fixed
    assert "//" not in fixed
    assert "lambda req, handler: handler(req)" in fixed


def test_inject_context_snippet_preserves_urls_and_strings():
    content = (
        "```python\n"
        'url = "https://example.com//docs"  // InjectContext docs link\n'
        "ctx = InjectContext()\n"
        "```"
    )
    update = _run(content)
    assert update is not None
    fixed = update["messages"][0].content
    assert '"https://example.com//docs"' in fixed
    assert "# InjectContext docs link" in fixed


def test_valid_python_fence_is_untouched():
    content = "```python\nx = 1  # comment\nf = lambda a: a + 1\n```"
    assert _run(content) is None


def test_javascript_fence_is_not_modified():
    content = "```javascript\nconst f = (a) => a + 1;  // arrow\n```"
    assert _run(content) is None
