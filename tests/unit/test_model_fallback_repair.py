"""Tests for tool-call history repair in the model fallback middleware."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.middleware.model_fallback_middleware import (
    MISSING_TOOL_RESULT_PLACEHOLDER,
    repair_tool_call_ids,
)


def _ai_with_tool_calls(*ids):
    return AIMessage(
        content="",
        tool_calls=[
            {"id": tc_id, "name": "search", "args": {}} for tc_id in ids
        ],
    )


def test_repair_inserts_missing_tool_message_when_partial():
    """An AIMessage with two tool_calls but only one ToolMessage gets one synthetic message."""
    messages = [
        HumanMessage(content="hi"),
        _ai_with_tool_calls("call_a", "call_b"),
        ToolMessage(tool_call_id="call_a", content="result a"),
        HumanMessage(content="next turn"),
    ]

    repaired = repair_tool_call_ids(messages)

    assert len(repaired) == len(messages) + 1
    inserted = repaired[3]
    assert isinstance(inserted, ToolMessage)
    assert inserted.tool_call_id == "call_b"
    assert inserted.content == MISSING_TOOL_RESULT_PLACEHOLDER
    assert isinstance(repaired[4], HumanMessage)


def test_repair_inserts_all_when_block_is_fully_dangling():
    """An AIMessage with tool_calls followed by no ToolMessages gets all synthetics."""
    messages = [
        _ai_with_tool_calls("call_x", "call_y"),
    ]

    repaired = repair_tool_call_ids(messages)

    assert len(repaired) == 3
    tool_msgs = [m for m in repaired if isinstance(m, ToolMessage)]
    assert {m.tool_call_id for m in tool_msgs} == {"call_x", "call_y"}
    for tm in tool_msgs:
        assert tm.content == MISSING_TOOL_RESULT_PLACEHOLDER


def test_repair_is_noop_when_all_tool_calls_responded():
    """A clean history with every tool_call_id answered must be unchanged."""
    messages = [
        HumanMessage(content="hi"),
        _ai_with_tool_calls("call_a", "call_b"),
        ToolMessage(tool_call_id="call_a", content="result a"),
        ToolMessage(tool_call_id="call_b", content="result b"),
        AIMessage(content="done"),
    ]

    repaired = repair_tool_call_ids(messages)

    assert len(repaired) == len(messages)
    for original, result in zip(messages, repaired):
        assert original is result


def test_repair_handles_multiple_tool_call_blocks():
    """Each AIMessage tool_calls block is repaired independently."""
    messages = [
        _ai_with_tool_calls("call_1"),
        # missing ToolMessage for call_1
        HumanMessage(content="continue"),
        _ai_with_tool_calls("call_2"),
        ToolMessage(tool_call_id="call_2", content="ok"),
    ]

    repaired = repair_tool_call_ids(messages)

    assert isinstance(repaired[1], ToolMessage)
    assert repaired[1].tool_call_id == "call_1"
    assert repaired[1].content == MISSING_TOOL_RESULT_PLACEHOLDER
    # second block should be untouched
    assert isinstance(repaired[3], AIMessage)
    assert isinstance(repaired[4], ToolMessage)
    assert repaired[4].tool_call_id == "call_2"


def test_repair_ignores_ai_message_without_tool_calls():
    """An AIMessage without tool_calls is left alone."""
    messages = [
        HumanMessage(content="hi"),
        AIMessage(content="just a reply"),
    ]

    repaired = repair_tool_call_ids(messages)

    assert repaired == messages
