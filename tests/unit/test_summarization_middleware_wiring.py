"""Tests for summarization middleware retry/fallback wiring."""

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agent import config
from src.middleware.summarization_middleware import CustomSummarizationMiddleware
from src.prompts.context_summary_prompt import context_summary_prompt


def test_summarization_middleware_uses_retrying_fallback_model():
    """Summarization should use the default model plus shared retry/fallback policy."""
    assert [model.id for model in config.FALLBACK_MODELS] == [
        "openai:gpt-5.4-nano",
        "anthropic:claude-haiku-4-5-20251001",
    ]

    middleware = CustomSummarizationMiddleware(
        model=config.DEFAULT_MODEL.id,
        summary_model=config.summarization_model,
        trigger=("tokens", 130_000),
        keep=("tokens", 30_000),
        summary_prompt=context_summary_prompt,
        trim_tokens_to_summarize=None,
    )

    summary_model = middleware.summary_model
    summary_primary = getattr(summary_model, "runnable", None)
    summary_fallbacks = getattr(summary_model, "fallbacks", [])

    assert middleware.model.model == "gemini-3.5-flash-lite"
    assert type(summary_model).__name__ == "RunnableWithFallbacks"
    assert getattr(summary_primary, "max_attempt_number") == config.MAX_RETRIES + 1
    assert len(summary_fallbacks) == len(config.FALLBACK_MODELS)
    assert all(
        getattr(fallback, "max_attempt_number") == config.MAX_RETRIES + 1
        for fallback in summary_fallbacks
    )


def test_stale_tool_results_are_elided_before_latest_human_message():
    """Only tool results from the current user turn retain their content."""
    middleware = CustomSummarizationMiddleware(
        model=FakeListChatModel(responses=["summary"]),
        summary_model=FakeListChatModel(responses=["summary"]),
        trigger=("tokens", 45_000),
        keep=("tokens", 12_000),
        summary_prompt=context_summary_prompt,
        trim_tokens_to_summarize=None,
    )
    tool_messages = [
        ToolMessage(content="a" * 20_000, name="search", tool_call_id=f"call-{index}")
        for index in range(4)
    ]
    messages = [
        HumanMessage(content="turn one"),
        AIMessage(content="answer one"),
        tool_messages[0],
        HumanMessage(content="turn two"),
        AIMessage(content="answer two"),
        tool_messages[1],
        HumanMessage(content="turn three"),
        AIMessage(content="answer three"),
        tool_messages[2],
        HumanMessage(content="latest turn"),
        AIMessage(content="latest answer"),
        tool_messages[3],
    ]

    result = middleware.before_model({"messages": messages}, None)

    assert result is not None
    updated_messages = result["messages"]
    assert [
        message.tool_call_id
        for message in updated_messages
        if isinstance(message, ToolMessage)
    ] == [
        "call-0",
        "call-1",
        "call-2",
        "call-3",
    ]
    assert [
        message.content
        for message in updated_messages
        if isinstance(message, ToolMessage)
    ] == [
        "[stale tool result elided: search, 20000 chars]",
        "[stale tool result elided: search, 20000 chars]",
        "[stale tool result elided: search, 20000 chars]",
        "a" * 20_000,
    ]
