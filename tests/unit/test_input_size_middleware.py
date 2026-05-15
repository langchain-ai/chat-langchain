"""Tests for InputSizeMiddleware.

Verifies the middleware short-circuits oversized single-turn user inputs
with a refusal AIMessage and a ``jump_to=end`` directive, while small
inputs pass through unchanged. Tests target ``before_agent`` directly so
they don't depend on the rest of the agent graph or any LLM calls.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from src.middleware.input_size_middleware import (
    DEFAULT_MAX_HUMAN_CHARS,
    REFUSAL_MESSAGE,
    InputSizeMiddleware,
)


def _state(messages):
    return {"messages": messages}


def test_oversized_single_message_short_circuits():
    middleware = InputSizeMiddleware()
    big = "x" * 30_000  # > 20K default threshold

    result = middleware.before_agent(_state([HumanMessage(content=big)]), runtime=None)

    assert isinstance(result, dict)
    assert result.get("jump_to") == "end"
    out_messages = result["messages"]
    assert len(out_messages) == 1
    assert isinstance(out_messages[0], AIMessage)
    assert out_messages[0].content == REFUSAL_MESSAGE


def test_small_message_passes_through():
    middleware = InputSizeMiddleware()
    small = "x" * 5_000  # well under threshold

    result = middleware.before_agent(_state([HumanMessage(content=small)]), runtime=None)

    assert result is None


def test_multi_turn_with_oversized_latest_short_circuits():
    middleware = InputSizeMiddleware()
    history = [
        HumanMessage(content="hi"),
        AIMessage(content="Hello! How can I help?"),
        HumanMessage(content="what's a runnable?"),
        AIMessage(content="A Runnable is the core abstraction..."),
        HumanMessage(content="x" * 30_000),  # latest user turn is oversized
    ]

    result = middleware.before_agent(_state(history), runtime=None)

    assert isinstance(result, dict)
    assert result.get("jump_to") == "end"
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert result["messages"][0].content == REFUSAL_MESSAGE


def test_default_threshold_is_20000():
    assert DEFAULT_MAX_HUMAN_CHARS == 20_000


def test_threshold_boundary_inclusive():
    # Exactly at threshold should pass through; one over should short-circuit.
    middleware = InputSizeMiddleware(max_human_chars=100)

    assert (
        middleware.before_agent(
            _state([HumanMessage(content="x" * 100)]), runtime=None
        )
        is None
    )

    over = middleware.before_agent(
        _state([HumanMessage(content="x" * 101)]), runtime=None
    )
    assert isinstance(over, dict)
    assert over.get("jump_to") == "end"


def test_non_human_last_message_passes_through():
    middleware = InputSizeMiddleware(max_human_chars=100)
    # Last message is an AIMessage — middleware should not interfere even
    # if its content is huge (it didn't come from the user).
    messages = [
        HumanMessage(content="hi"),
        AIMessage(content="x" * 10_000),
    ]
    assert middleware.before_agent(_state(messages), runtime=None) is None


def test_empty_messages_passes_through():
    middleware = InputSizeMiddleware()
    assert middleware.before_agent(_state([]), runtime=None) is None


def test_list_content_blocks_are_measured():
    middleware = InputSizeMiddleware(max_human_chars=100)
    blocks = [
        {"type": "text", "text": "x" * 60},
        {"type": "text", "text": "y" * 60},
    ]
    msg = HumanMessage(content=blocks)
    result = middleware.before_agent(_state([msg]), runtime=None)
    assert isinstance(result, dict)
    assert result.get("jump_to") == "end"


def test_custom_threshold_argument():
    middleware = InputSizeMiddleware(max_human_chars=50)
    result = middleware.before_agent(
        _state([HumanMessage(content="x" * 60)]), runtime=None
    )
    assert isinstance(result, dict)
    assert result.get("jump_to") == "end"


def test_env_var_threshold(monkeypatch):
    monkeypatch.setenv("MAX_HUMAN_CHARS", "75")
    middleware = InputSizeMiddleware()
    assert middleware.max_human_chars == 75
    assert (
        middleware.before_agent(
            _state([HumanMessage(content="x" * 75)]), runtime=None
        )
        is None
    )
    result = middleware.before_agent(
        _state([HumanMessage(content="x" * 76)]), runtime=None
    )
    assert isinstance(result, dict)
    assert result.get("jump_to") == "end"


def test_env_var_invalid_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("MAX_HUMAN_CHARS", "not-a-number")
    middleware = InputSizeMiddleware()
    assert middleware.max_human_chars == DEFAULT_MAX_HUMAN_CHARS
