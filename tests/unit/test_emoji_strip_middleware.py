"""Tests for the egress emoji guard on the docs agent's final answer."""

from __future__ import annotations

import os
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

os.environ["USE_LOCAL_PROMPTS"] = "1"

from src.middleware.emoji_strip_middleware import EmojiStripMiddleware, strip_emoji


def test_after_agent_strips_emoji_from_english_answer():
    middleware = EmojiStripMiddleware()
    ai = AIMessage(
        content=(
            "**Bind tools to your LLM.** 💡\n\n"
            "- ✅ Done\n"
            "- ❌ Not supported\n"
            "- 🛠 Use `bind_tools()` 👈\n"
        ),
        id="a1",
    )
    state = {"messages": [HumanMessage(content="how?"), ai]}

    update = middleware.after_agent(state, runtime=SimpleNamespace())

    assert update is not None
    assert update["messages"][0].id == "a1"
    content = update["messages"][0].content
    assert "💡" not in content
    assert "- Done\n" in content
    assert "- Not supported\n" in content
    assert "- Use `bind_tools()`" in content


def test_after_agent_strips_emoji_from_korean_answer():
    middleware = EmojiStripMiddleware()
    ai = AIMessage(content="**도구를 바인딩하세요.** ✅\n\n- 💡 팁: `bind_tools()`\n", id="a1")
    state = {"messages": [ai]}

    update = middleware.after_agent(state, runtime=SimpleNamespace())

    assert update is not None
    content = update["messages"][0].content
    assert "✅" not in content and "💡" not in content
    assert "**도구를 바인딩하세요.**" in content
    assert "- 팁: `bind_tools()`" in content


def test_code_blocks_and_inline_code_are_preserved_verbatim():
    text = (
        "Here is your code 🚀\n\n"
        "```python\n"
        "# 🚀 launch  it\n"
        'print("✅  ok")\n'
        "```\n\n"
        "Inline `emoji = '✅'` stays too. ❌\n"
    )

    cleaned = strip_emoji(text)

    assert "# 🚀 launch  it" in cleaned
    assert 'print("✅  ok")' in cleaned
    assert "`emoji = '✅'`" in cleaned
    assert "Here is your code \n" in cleaned
    assert "stays too. \n" in cleaned


def test_emoji_free_answer_is_returned_unchanged():
    middleware = EmojiStripMiddleware()
    content = "**Use `bind_tools()`.**\n\n- One\n- Two\n"
    state = {"messages": [AIMessage(content=content, id="a1")]}

    assert middleware.after_agent(state, runtime=SimpleNamespace()) is None
    assert state["messages"][0].content == content


def test_markdown_structure_survives_stripping():
    middleware = EmojiStripMiddleware()
    ai = AIMessage(
        content=(
            "**Configure `langgraph.json` to set a TTL.** ✅\n\n"
            "## Store Item TTL 🚀\n\n"
            "Use `refresh_on_read` to reset timers.\n\n"
            "**Relevant docs:**\n\n"
            "- [TTL Configuration Guide](https://docs.langchain.com/configure-ttl)\n"
        ),
        id="a1",
    )
    state = {"messages": [ai]}

    update = middleware.after_agent(state, runtime=SimpleNamespace())

    content = update["messages"][0].content
    assert content.startswith("**Configure `langgraph.json` to set a TTL.**")
    assert "## Store Item TTL" in content
    assert "`refresh_on_read`" in content
    assert (
        "- [TTL Configuration Guide](https://docs.langchain.com/configure-ttl)"
        in content
    )
    assert "✅" not in content and "🚀" not in content


def test_text_content_blocks_are_cleaned():
    middleware = EmojiStripMiddleware()
    ai = AIMessage(
        content=[{"type": "text", "text": "All set ✅"}, {"type": "other", "data": 1}],
        id="a1",
    )
    state = {"messages": [ai]}

    update = middleware.after_agent(state, runtime=SimpleNamespace())

    assert update["messages"][0].content[0]["text"] == "All set "
    assert update["messages"][0].content[1] == {"type": "other", "data": 1}
