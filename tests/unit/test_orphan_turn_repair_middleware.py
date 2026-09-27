"""Tests for repairing answerless checkpointed turns."""

from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.graph.message import add_messages

from src.middleware.orphan_turn_repair_middleware import OrphanTurnRepairMiddleware


def test_before_agent_repairs_orphaned_turn_before_new_human_message():
    orphan_human = HumanMessage(content="First question", id="human-old")
    residue = [
        message
        for index in range(30)
        for message in (
            AIMessage(content="", id=f"ai-{index}"),
            ToolMessage(
                content="same result",
                name="check_links",
                tool_call_id=f"tool-call-{index}",
                id=f"tool-{index}",
            ),
        )
    ]
    new_human = HumanMessage(content="New question", id="human-new")
    state = {"messages": [orphan_human, *residue, new_human]}

    update = OrphanTurnRepairMiddleware().before_agent(state, runtime=SimpleNamespace())
    repaired = add_messages(state["messages"], update["messages"])

    assert len([message for message in repaired if isinstance(message, AIMessage)]) == 1
    assert repaired[-3].content == "First question"
    assert (
        repaired[-2].content == "The previous turn failed before producing an answer."
    )
    assert isinstance(repaired[-1], HumanMessage)
    assert repaired[-1].content == "New question"
    assert all(not isinstance(message, RemoveMessage) for message in repaired)
