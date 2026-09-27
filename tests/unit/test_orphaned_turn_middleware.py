"""Tests for failed-turn history cleanup."""

from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.middleware.orphaned_turn_middleware import OrphanedTurnMiddleware


class _Callbacks:
    def __init__(self):
        self.metadata = []

    def add_metadata(self, metadata):
        self.metadata.append(metadata)


def test_before_model_removes_orphaned_turn_and_repeated_tool_messages(monkeypatch):
    callbacks = _Callbacks()
    monkeypatch.setattr(
        "src.middleware.orphaned_turn_middleware.get_config",
        lambda: {"callbacks": callbacks},
    )
    messages = [
        HumanMessage(content="Unanswered question", id="human-1"),
        AIMessage(content="", id="ai-empty"),
    ]
    for index in range(32):
        messages.extend(
            [
                AIMessage(
                    content="",
                    id=f"tool-call-{index}",
                    tool_calls=[
                        {
                            "name": "check_links",
                            "args": {"url": "https://example.com"},
                            "id": f"call-{index}",
                            "type": "tool_call",
                        }
                    ],
                ),
                ToolMessage(
                    content="",
                    id=f"tool-result-{index}",
                    tool_call_id=f"call-{index}",
                ),
            ]
        )
    messages.append(HumanMessage(content="Current question", id="human-2"))

    update = OrphanedTurnMiddleware().before_model(
        {"messages": messages}, runtime=SimpleNamespace()
    )

    retained = update["messages"][1:]
    assert [message.id for message in retained] == ["human-2"]
    assert callbacks.metadata == [
        {"orphaned_turn_stripped": True, "orphaned_turn_count": 1}
    ]


def test_before_model_keeps_completed_alternating_history(monkeypatch):
    callbacks = _Callbacks()
    monkeypatch.setattr(
        "src.middleware.orphaned_turn_middleware.get_config",
        lambda: {"callbacks": callbacks},
    )
    messages = [
        HumanMessage(content="First question", id="human-1"),
        AIMessage(content="First answer", id="ai-1"),
        HumanMessage(content="Second question", id="human-2"),
        AIMessage(content="Second answer", id="ai-2"),
    ]

    assert (
        OrphanedTurnMiddleware().before_model(
            {"messages": messages}, runtime=SimpleNamespace()
        )
        is None
    )
    assert callbacks.metadata == []
