from types import SimpleNamespace

from langchain_core.messages import AIMessage

from src.middleware.final_answer_guard_middleware import FinalAnswerGuardMiddleware


def test_tool_call_residue_is_stripped_when_structured_calls_are_present():
    message = AIMessage(
        content=[
            {
                "args": {"query": "LangChain"},
                "id": "call-1",
                "name": "search",
                "type": "tool_call",
            }
        ],
        id="message-1",
        name="assistant",
        response_metadata={"finish_reason": "tool_calls"},
        tool_calls=[
            {
                "args": {"query": "LangChain"},
                "id": "call-1",
                "name": "search",
                "type": "tool_call",
            }
        ],
    )
    state = {"messages": [message]}

    update = FinalAnswerGuardMiddleware().after_model(state, SimpleNamespace())

    sanitized = update["messages"][0]
    assert sanitized.content == ""
    assert sanitized.id == message.id
    assert sanitized.name == message.name
    assert sanitized.response_metadata == message.response_metadata
    assert sanitized.tool_calls == message.tool_calls
    assert "jump_to" not in update


def test_malformed_tool_call_content_retries_then_ends_with_safe_message():
    middleware = FinalAnswerGuardMiddleware()
    malformed = AIMessage.model_construct(
        content={"args": {"query": "LangChain"}, "name": "search"},
        id="message-1",
    )

    first_update = middleware.after_model({"messages": [malformed]}, SimpleNamespace())
    assert first_update["messages"][0].content == ""
    assert first_update["final_answer_guard_retried"] is True
    assert first_update["jump_to"] == "model"

    retry_message = AIMessage.model_construct(
        content=malformed.content,
        id=malformed.id,
    )
    second_update = middleware.after_model(
        {
            "messages": [retry_message],
            "final_answer_guard_retried": True,
        },
        SimpleNamespace(),
    )
    assert second_update["messages"][0].content == (
        "I couldn't produce a readable answer. Please resend your question."
    )
    assert second_update["jump_to"] == "end"


def test_markdown_prose_is_unchanged():
    content = "## Answer\n\nUse `create_agent` to build the workflow."
    message = AIMessage(content=content)

    update = FinalAnswerGuardMiddleware().after_model(
        {"messages": [message]}, SimpleNamespace()
    )

    assert update is None
    assert message.content == content
