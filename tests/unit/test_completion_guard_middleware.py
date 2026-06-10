"""Tests for the EnsureCompletionMiddleware truncation detector."""

from langchain_core.messages import AIMessage, HumanMessage

from src.middleware.completion_guard_middleware import (
    EnsureCompletionMiddleware,
    is_truncated,
)


def test_detects_unclosed_python_fence():
    text = (
        "Here is the example you asked for:\n\n"
        "```python\n"
        "def hello():\n"
        "    print('hi')\n"
    )
    assert is_truncated(text) is True


def test_detects_unclosed_curl_header_line():
    # Real-world truncation: open code fence AND open shell arg quote at EOF.
    text = (
        "Run this curl request:\n\n"
        "```bash\n"
        "curl https://api.example.com \\\n"
        '  -H "Authorization: Bearer abc" \\\n'
        '  -H "'
    )
    assert is_truncated(text) is True


def test_detects_open_curl_header_with_balanced_fences():
    # Even if fences happened to balance, an open `-H "` shell arg is a tell.
    text = (
        'curl https://api.example.com \\\n'
        '  -H "Authorization: Bearer abc" \\\n'
        '  -H "'
    )
    assert is_truncated(text) is True


def test_properly_terminated_answer_does_not_trigger():
    text = (
        "Here you go:\n\n"
        "```python\n"
        "print('hello world')\n"
        "```\n\n"
        "Let me know if you want anything else."
    )
    assert is_truncated(text) is False


def test_middleware_injects_continue_turn_on_truncation():
    middleware = EnsureCompletionMiddleware()
    state = {
        "messages": [
            HumanMessage(content="Give me a python example"),
            AIMessage(content="```python\ndef hello():\n    print('hi')\n"),
        ]
    }

    result = middleware.after_model(state, runtime=None)  # type: ignore[arg-type]

    assert result is not None
    new_messages = result["messages"]
    assert len(new_messages) == 1
    assert isinstance(new_messages[0], HumanMessage)
    assert new_messages[0].additional_kwargs.get("completion_guard") is True


def test_middleware_noop_on_complete_response():
    middleware = EnsureCompletionMiddleware()
    state = {
        "messages": [
            HumanMessage(content="hi"),
            AIMessage(content="All done. Let me know if you need anything else."),
        ]
    }

    assert middleware.after_model(state, runtime=None) is None  # type: ignore[arg-type]


def test_middleware_does_not_loop_past_max_continuations():
    middleware = EnsureCompletionMiddleware(max_continuations=1)
    state = {
        "messages": [
            HumanMessage(content="give me code"),
            AIMessage(content="```python\nx = 1\n"),
            HumanMessage(
                content="Please continue",
                additional_kwargs={"completion_guard": True},
            ),
            AIMessage(content="```python\ny = 2\n"),
        ]
    }

    assert middleware.after_model(state, runtime=None) is None  # type: ignore[arg-type]


def test_middleware_skips_messages_with_pending_tool_calls():
    middleware = EnsureCompletionMiddleware()
    tool_call_message = AIMessage(
        content="```python\nincomplete",
        tool_calls=[
            {"name": "search", "args": {"q": "x"}, "id": "call_1", "type": "tool_call"}
        ],
    )
    state = {"messages": [HumanMessage(content="hi"), tool_call_message]}

    assert middleware.after_model(state, runtime=None) is None  # type: ignore[arg-type]
