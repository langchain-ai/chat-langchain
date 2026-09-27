"""Tests for failed-turn recovery middleware."""

import asyncio

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.middleware.agent_failure_recovery import (
    MAX_CHECK_LINKS_CALLS,
    RECOVERY_MESSAGE,
    AgentFailureRecoveryMiddleware,
)


def _loop_messages(count: int) -> list[object]:
    messages: list[object] = [HumanMessage(content="Check these links.")]
    for index in range(count):
        messages.extend(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "check_links",
                            "args": {"urls": ["https://example.com"]},
                            "id": f"call-{index}",
                            "type": "tool_call",
                        }
                    ],
                ),
                ToolMessage(content="checked", tool_call_id=f"call-{index}"),
            ]
        )
    return messages


def test_before_model_ends_repeating_check_links_loop():
    middleware = AgentFailureRecoveryMiddleware()

    update = middleware.before_model(
        {"messages": _loop_messages(MAX_CHECK_LINKS_CALLS)}, runtime=None
    )

    assert update["jump_to"] == "end"
    assert update["messages"][0].content == RECOVERY_MESSAGE


def test_before_model_leaves_normal_turn_unchanged():
    middleware = AgentFailureRecoveryMiddleware()

    assert (
        middleware.before_model({"messages": _loop_messages(1)}, runtime=None) is None
    )


def test_model_failure_returns_terminal_message():
    middleware = AgentFailureRecoveryMiddleware()

    async def fail(_request):
        raise RuntimeError("model unavailable")

    result = asyncio.run(middleware.awrap_model_call(None, fail))

    assert result.result[0].content == RECOVERY_MESSAGE
