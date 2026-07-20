"""Tests for the egress link allowlist middleware."""

from __future__ import annotations

import os
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

os.environ["USE_LOCAL_PROMPTS"] = "1"

from src.middleware.link_allowlist_middleware import (
    LinkAllowlistMiddleware,
    _strip_disallowed_links,
)


def test_strips_third_party_documentation_link():
    text = "See [Agent Skills standard](https://agentskills.io/specification) for details."
    assert _strip_disallowed_links(text) == "See Agent Skills standard for details."


def test_keeps_langchain_and_subdomain_links():
    text = (
        "[Docs](https://docs.langchain.com/oss/python/streaming) and "
        "[Home](https://www.langchain.com/pricing) and "
        "[Root](https://langchain.com)"
    )
    assert _strip_disallowed_links(text) == text


def test_does_not_match_lookalike_domain():
    text = "[Fake](https://notlangchain.com/docs)"
    assert _strip_disallowed_links(text) == "Fake"


def test_after_agent_rewrites_final_ai_message():
    middleware = LinkAllowlistMiddleware()
    answer = "Use [the spec](https://agentskills.io/specification)."
    ai = AIMessage(content=answer, id="a1")
    state = {"messages": [HumanMessage(content="skills?"), ai]}

    update = middleware.after_agent(state, runtime=SimpleNamespace())

    assert update is not None
    assert update["messages"][0].id == "a1"
    assert update["messages"][0].content == "Use the spec."


def test_after_agent_noop_when_all_links_allowed():
    middleware = LinkAllowlistMiddleware()
    ai = AIMessage(
        content="[Docs](https://docs.langchain.com/oss)", id="a1"
    )
    state = {"messages": [HumanMessage(content="q"), ai]}

    assert middleware.after_agent(state, runtime=SimpleNamespace()) is None


def test_after_agent_handles_list_content_blocks():
    middleware = LinkAllowlistMiddleware()
    ai = AIMessage(
        content=[
            {"type": "text", "text": "See [x](https://agentskills.io/)."},
        ],
        id="a1",
    )
    state = {"messages": [HumanMessage(content="q"), ai]}

    update = middleware.after_agent(state, runtime=SimpleNamespace())

    assert update is not None
    assert update["messages"][0].content[0]["text"] == "See x."
