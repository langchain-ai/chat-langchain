"""Tests for turn-safe summarization reconstruction."""

import asyncio

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.middleware.summarization_middleware import CustomSummarizationMiddleware
from src.utils.message_utils import latest_user_message_index


def _middleware() -> CustomSummarizationMiddleware:
    middleware = object.__new__(CustomSummarizationMiddleware)
    middleware._ensure_message_ids = lambda messages: None
    middleware.token_counter = lambda messages: 1
    middleware._should_summarize = lambda messages, total_tokens: True
    middleware._determine_cutoff_index = lambda messages: 3
    middleware._create_summary = lambda messages: "summary"

    async def create_async_summary(messages: list) -> str:
        return "async summary"

    middleware._acreate_summary = create_async_summary
    return middleware


def _messages() -> list:
    return [
        HumanMessage(content="Earlier question"),
        HumanMessage(content="Current question"),
        AIMessage(
            content="",
            tool_calls=[{"name": "search", "args": {"query": "docs"}, "id": "call-1"}],
        ),
        ToolMessage(content="First result", tool_call_id="call-1"),
        ToolMessage(content="Second result", tool_call_id="call-1"),
        ToolMessage(content="Third result", tool_call_id="call-1"),
    ]


def _assert_reconstruction(result: dict, original_messages: list) -> None:
    reconstructed = result["messages"]
    assert isinstance(reconstructed[1], SystemMessage)
    assert reconstructed[1].additional_kwargs["lc_source"] == "summarization"
    assert not any(
        isinstance(message, HumanMessage)
        and message.additional_kwargs.get("lc_source") == "summarization"
        for message in reconstructed
    )
    assert reconstructed[2:] == original_messages[1:]
    assert latest_user_message_index(reconstructed) == 2


def test_before_model_preserves_current_turn_tool_results() -> None:
    messages = _messages()
    result = _middleware().before_model({"messages": messages}, None)

    assert result is not None
    _assert_reconstruction(result, messages)


def test_abefore_model_preserves_current_turn_tool_results() -> None:
    messages = _messages()
    result = asyncio.run(_middleware().abefore_model({"messages": messages}, None))

    assert result is not None
    _assert_reconstruction(result, messages)
