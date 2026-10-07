"""Tests for abandoned-turn cleanup under the real messages reducer."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph.message import add_messages

from src.middleware.abandoned_turn_middleware import AbandonedTurnMiddleware
from src.prompts.docs_agent_prompt import docs_agent_prompt


def _run_middleware(messages):
    update = AbandonedTurnMiddleware().before_agent(
        {"messages": messages}, runtime=SimpleNamespace()
    )
    return add_messages(messages, update["messages"]) if update else messages


def _assert_only_latest_unanswered(messages):
    human_indices = [
        index
        for index, message in enumerate(messages)
        if isinstance(message, HumanMessage)
    ]
    unanswered = []
    for position, start in enumerate(human_indices):
        end = (
            human_indices[position + 1]
            if position + 1 < len(human_indices)
            else len(messages)
        )
        if not any(
            isinstance(message, AIMessage)
            and message.text.strip()
            and not message.tool_calls
            for message in messages[start + 1 : end]
        ):
            unanswered.append(messages[start].id)
    assert unanswered == [messages[human_indices[-1]].id]


@pytest.mark.parametrize(
    "partial_replies",
    [
        pytest.param([], id="interrupted"),
        pytest.param([AIMessage(content="", id="rolled-back")], id="rollback"),
        pytest.param(
            [
                AIMessage(
                    content=[{"type": "reasoning", "reasoning": "thinking"}],
                    id="thinking",
                )
            ],
            id="reasoning-only",
        ),
    ],
)
def test_abandoned_turn_is_closed(partial_replies):
    previous = HumanMessage(content="Cancelled question", id="previous")
    latest = HumanMessage(content="Current question", id="latest")
    messages = [previous, *partial_replies, latest]

    result = _run_middleware(messages)

    assert result[0] == previous
    assert result[1].id == "abandoned-previous"
    assert result[1].content == (
        "(This earlier request was cancelled before an answer was produced.)"
    )
    assert result[-1] == latest
    assert messages == [previous, *partial_replies, latest]
    _assert_only_latest_unanswered(result)


def test_recursion_killed_turn_removes_tool_pairs_and_preserves_latest_turn():
    previous = HumanMessage(content="Cancelled question", id="previous")
    latest = HumanMessage(content="Current question", id="latest")
    latest_call = AIMessage(
        content="Checking current links",
        id="latest-call",
        tool_calls=[{"name": "check_links", "args": {}, "id": "current"}],
    )
    latest_result = ToolMessage(
        content="valid", tool_call_id="current", id="latest-result"
    )
    messages = [
        previous,
        AIMessage(
            content="Checking links",
            id="orphan-call",
            tool_calls=[{"name": "check_links", "args": {}, "id": "cancelled"}],
        ),
        ToolMessage(content="invalid", tool_call_id="cancelled", id="orphan-result"),
        AIMessage(
            content="",
            id="unfinished-call",
            tool_calls=[{"name": "check_links", "args": {}, "id": "unfinished"}],
        ),
        latest,
        latest_call,
        latest_result,
    ]

    result = _run_middleware(messages)

    assert [message.id for message in result] == [
        "previous",
        "abandoned-previous",
        "latest",
        "latest-call",
        "latest-result",
    ]
    assert result[-3:] == [latest, latest_call, latest_result]
    _assert_only_latest_unanswered(result)


@pytest.mark.parametrize(
    "answer",
    [
        AIMessage(content="Answered", id="answer"),
        AIMessage(content=[{"type": "text", "text": "Answered"}], id="answer"),
    ],
)
def test_clean_history_returns_none(answer):
    messages = [
        HumanMessage(content="Answered question", id="previous"),
        answer,
        HumanMessage(content="Current question", id="latest"),
    ]

    assert (
        AbandonedTurnMiddleware().before_agent(
            {"messages": messages}, runtime=SimpleNamespace()
        )
        is None
    )
    _assert_only_latest_unanswered(messages)


def test_multiple_abandoned_turns_preserve_clean_history_and_are_idempotent():
    prefix = [
        SystemMessage(content="Instructions", id="system"),
        HumanMessage(content="Answered question", id="answered"),
        AIMessage(content="Answer", id="answer"),
    ]
    messages = [
        *prefix,
        HumanMessage(content="First cancelled question", id="first"),
        HumanMessage(content="Second cancelled question", id="second"),
        HumanMessage(content="Current question", id="latest"),
    ]

    result = _run_middleware(messages)

    assert result[:3] == prefix
    assert [message.id for message in result[3:]] == [
        "first",
        "abandoned-first",
        "second",
        "abandoned-second",
        "latest",
    ]
    _assert_only_latest_unanswered(result)
    assert (
        AbandonedTurnMiddleware().before_agent(
            {"messages": result}, runtime=SimpleNamespace()
        )
        is None
    )
    _assert_only_latest_unanswered(_run_middleware(result))


def test_newest_turn_and_its_pending_tool_call_are_untouched():
    messages = [
        HumanMessage(content="Current question", id="latest"),
        AIMessage(
            content="",
            id="pending-call",
            tool_calls=[{"name": "check_links", "args": {}, "id": "pending"}],
        ),
    ]

    assert (
        AbandonedTurnMiddleware().before_agent(
            {"messages": messages}, runtime=SimpleNamespace()
        )
        is None
    )
    _assert_only_latest_unanswered(messages)


@pytest.mark.asyncio
async def test_cleanup_runs_in_async_agent():
    agent = create_agent(
        FakeListChatModel(responses=["Current answer"]),
        middleware=[AbandonedTurnMiddleware()],
    )
    result = await agent.ainvoke(
        {
            "messages": [
                HumanMessage(content="Cancelled question", id="previous"),
                HumanMessage(content="Current question", id="latest"),
            ]
        }
    )

    assert [message.id for message in result["messages"][:-1]] == [
        "previous",
        "abandoned-previous",
        "latest",
    ]
    assert result["messages"][-1].content == "Current answer"
    _assert_only_latest_unanswered(result["messages"][:-1])


def test_middleware_order_and_active_prompt_rule():
    root = Path(__file__).resolve().parents[2]
    module = ast.parse((root / "agent.py").read_text())
    middleware = next(
        statement.value
        for statement in module.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "docs_agent_middleware"
            for target in statement.targets
        )
    )
    assert [entry.func.id for entry in middleware.elts[:4]] == [
        "IngressGuardsMiddleware",
        "AbandonedTurnMiddleware",
        "GuardrailsMiddleware",
        "CustomSummarizationMiddleware",
    ]
    rule = (
        "Answer only the most recent user message; earlier user messages without an "
        "answer were cancelled and must not be answered unless the latest message refers to them."
    )
    assert rule in (root / "instructions.md").read_text()
    assert rule in docs_agent_prompt
