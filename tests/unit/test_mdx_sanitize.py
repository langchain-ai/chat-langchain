"""Tests for stripping Mintlify/MDX authoring artifacts from docs tool output."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from langchain_core.messages import ToolMessage

from src.middleware.mdx_sanitize_middleware import MdxSanitizeMiddleware
from src.utils.mdx_sanitize import strip_mdx_artifacts


def test_strips_code_highlight_annotation():
    source = "checkpointer=checkpointer  # [!code highlight]"

    assert strip_mdx_artifacts(source) == "checkpointer=checkpointer"


def test_strips_all_code_annotation_flavors():
    source = "\n".join(
        [
            "graph = builder.compile() # [!code focus]",
            "const x = 1; // [!code ++]",
            "const y = 2; // [!code --]",
            "value = 3 [!code highlight]",
        ]
    )

    cleaned = strip_mdx_artifacts(source)

    assert "[!code" not in cleaned
    assert cleaned.splitlines() == [
        "graph = builder.compile()",
        "const x = 1;",
        "const y = 2;",
        "value = 3",
    ]


def test_strips_code_fence_theme_metadata():
    source = '```python Google theme={"theme":{"light":"catppuccin-latte"}}'

    cleaned = strip_mdx_artifacts(source)

    assert "theme=" not in cleaned
    assert cleaned == "```python Google"


def test_unwraps_mintlify_component_keeping_inner_content():
    source = "<Warning>\n  LocalShellBackend executes arbitrary code\n</Warning>"

    cleaned = strip_mdx_artifacts(source)

    assert "<Warning>" not in cleaned
    assert "</Warning>" not in cleaned
    assert cleaned.strip() == "LocalShellBackend executes arbitrary code"


def test_unwraps_components_with_attributes():
    source = '<Tabs>\n<Tab title="Python">\ncontent\n</Tab>\n</Tabs>'

    cleaned = strip_mdx_artifacts(source)

    assert "<Tab" not in cleaned
    assert cleaned.strip() == "content"


def test_absolutizes_root_relative_docs_links():
    source = "[Event streaming](/oss/python/langchain/event-streaming)"

    assert strip_mdx_artifacts(source) == (
        "[Event streaming](https://docs.langchain.com/oss/python/langchain/event-streaming)"
    )


def test_absolutizes_langsmith_and_labs_links():
    source = "[Tracing](/langsmith/observability) and [Labs](/labs/overview)"

    cleaned = strip_mdx_artifacts(source)

    assert "](/langsmith/" not in cleaned
    assert "](/labs/" not in cleaned
    assert "https://docs.langchain.com/langsmith/observability" in cleaned
    assert "https://docs.langchain.com/labs/overview" in cleaned


def test_leaves_clean_markdown_untouched():
    source = "**Use a checkpointer.**\n\n```python\ngraph.compile()\n```\n"

    assert strip_mdx_artifacts(source) == source


def _tool_request(name: str) -> SimpleNamespace:
    return SimpleNamespace(tool_call={"name": name, "id": "call-1"})


def test_middleware_sanitizes_docs_tool_result():
    middleware = MdxSanitizeMiddleware()
    request = _tool_request("query_docs_filesystem_docs_by_lang_chain")
    raw = "<Tip>\nSee [streaming](/oss/python/langgraph/streaming)\n</Tip>"

    def handler(req):
        return ToolMessage(content=raw, tool_call_id="call-1")

    result = asyncio.run(middleware.awrap_tool_call(request, _async(handler)))

    assert "<Tip>" not in result.content
    assert "https://docs.langchain.com/oss/python/langgraph/streaming" in result.content


def test_middleware_ignores_other_tools():
    middleware = MdxSanitizeMiddleware()
    request = _tool_request("get_support_article_content")
    raw = "value  # [!code highlight]"

    result = middleware.wrap_tool_call(
        request, lambda req: ToolMessage(content=raw, tool_call_id="call-1")
    )

    assert result.content == raw


def _async(handler):
    async def _handler(request):
        return handler(request)

    return _handler
