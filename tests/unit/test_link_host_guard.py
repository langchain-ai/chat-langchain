"""Tests for hostname allowlisting in link validation and the output guard."""

import asyncio
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

from src.middleware.link_host_guard_middleware import LinkHostGuardMiddleware
from src.tools.link_check_tools import (
    UNAPPROVED_HOST_ERROR,
    _cache,
    _check_single_url,
)


def _check(url: str):
    _cache.pop(url, None)
    # client is never touched: unapproved hosts are rejected before any request.
    return asyncio.run(_check_single_url(client=None, url=url, timeout=1.0))


def test_check_links_rejects_lookalike_host_without_network():
    result = _check("https://docs.linkchain.com/oss/javascript/integrations/chat/openai")

    assert result.valid is False
    assert result.error == UNAPPROVED_HOST_ERROR
    assert result.status_code is None


def test_check_links_rejects_unowned_langsmith_docs_host():
    result = _check("https://docs.langsmith.com/admin")

    assert result.valid is False
    assert result.error == UNAPPROVED_HOST_ERROR


def test_middleware_repairs_unapproved_host_on_docs_path():
    middleware = LinkHostGuardMiddleware()
    answer = AIMessage(
        content=(
            "See [image generation]"
            "(https://docs.linkchain.com/oss/javascript/integrations/chat/openai#image-generation)"
        ),
        id="a1",
    )
    state = {"messages": [HumanMessage(content="hi"), answer]}

    update = middleware.after_model(state, runtime=SimpleNamespace())

    assert update is not None
    assert update["messages"][0].id == "a1"
    assert update["messages"][0].content == (
        "See [image generation]"
        "(https://docs.langchain.com/oss/javascript/integrations/chat/openai#image-generation)"
    )


def test_middleware_leaves_approved_hosts_untouched():
    middleware = LinkHostGuardMiddleware()
    answer = AIMessage(
        content="Open [LangSmith](https://smith.langchain.com) to view traces.",
        id="a1",
    )
    state = {"messages": [answer]}

    assert middleware.after_model(state, runtime=SimpleNamespace()) is None


def test_middleware_drops_unapproved_link_keeping_text():
    middleware = LinkHostGuardMiddleware()
    answer = AIMessage(content="Read [the guide](https://evil.example.com/guide).", id="a1")
    state = {"messages": [answer]}

    update = middleware.after_model(state, runtime=SimpleNamespace())

    assert update is not None
    assert update["messages"][0].content == "Read the guide."


def test_middleware_sanitizes_text_content_blocks():
    middleware = LinkHostGuardMiddleware()
    answer = AIMessage(
        content=[
            {"type": "text", "text": "https://docs.linkchain.com/langsmith/admin"},
            {"type": "text", "text": "https://docs.langchain.com/langsmith/admin"},
        ],
        id="a1",
    )
    state = {"messages": [answer]}

    update = middleware.after_model(state, runtime=SimpleNamespace())

    assert update is not None
    assert update["messages"][0].content[0]["text"] == (
        "https://docs.langchain.com/langsmith/admin"
    )
