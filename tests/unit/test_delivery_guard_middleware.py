"""Tests for the unattended-run delivery guard."""

from __future__ import annotations

import os
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

os.environ["USE_LOCAL_PROMPTS"] = "1"

from src.middleware import delivery_guard_middleware
from src.middleware.delivery_guard_middleware import DeliveryGuardMiddleware


def _runtime(source: str | None = None):
    return SimpleNamespace(context={"run_source": source} if source else {})


def _write_file_call(path: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "write_file", "args": {"file_path": path}, "id": "call_write"}
        ],
    )


def _record_interrupts(monkeypatch) -> list[dict]:
    payloads: list[dict] = []
    monkeypatch.setattr(
        delivery_guard_middleware,
        "interrupt",
        lambda payload: payloads.append(payload),
    )
    return payloads


def test_trigger_run_writing_only_to_sandbox_does_not_claim_success(monkeypatch):
    payloads = _record_interrupts(monkeypatch)
    middleware = DeliveryGuardMiddleware()
    state = {
        "messages": [
            HumanMessage(content="Scan Gmail and create a live page"),
            _write_file_call("/tmp/allwhere_in_transit_status.html"),
            AIMessage(
                content="Done - I built a live status page here: /tmp/allwhere_in_transit_status.html",
                id="ai-final",
            ),
        ]
    }

    update = middleware.after_model(state, _runtime("cron"))

    assert update is not None
    final = update["messages"][0]
    assert final.id == "ai-final"
    assert "live status page" not in final.content
    assert "/tmp/allwhere_in_transit_status.html" in final.content
    assert payloads and payloads[0]["type"] == "message_user"
    assert payloads[0]["reason"] == "missing_delivery_destination"
    assert payloads[0]["sandbox_artifacts"] == ["/tmp/allwhere_in_transit_status.html"]


def test_trigger_run_that_delivered_is_left_alone(monkeypatch):
    payloads = _record_interrupts(monkeypatch)
    middleware = DeliveryGuardMiddleware()
    state = {
        "messages": [
            HumanMessage(content="Scan Gmail and post the digest"),
            _write_file_call("/tmp/digest.html"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "slack_post_message",
                        "args": {"channel": "#it-ops"},
                        "id": "call_slack",
                    }
                ],
            ),
            AIMessage(content="Posted the digest to #it-ops.", id="ai-final"),
        ]
    }

    assert middleware.after_model(state, _runtime("trigger")) is None
    assert payloads == []


def test_interactive_run_is_not_guarded(monkeypatch):
    payloads = _record_interrupts(monkeypatch)
    middleware = DeliveryGuardMiddleware()
    state = {
        "messages": [
            HumanMessage(content="Scan Gmail and create a live page"),
            _write_file_call("/tmp/digest.html"),
            AIMessage(content="Done - I created the page.", id="ai-final"),
        ]
    }

    assert middleware.after_model(state, _runtime()) is None
    assert payloads == []


def test_trigger_run_still_calling_tools_is_not_guarded(monkeypatch):
    payloads = _record_interrupts(monkeypatch)
    middleware = DeliveryGuardMiddleware()
    state = {"messages": [_write_file_call("/tmp/digest.html")]}

    assert middleware.after_model(state, _runtime("schedule")) is None
    assert payloads == []


def test_trigger_run_without_artifacts_or_claim_is_not_guarded(monkeypatch):
    payloads = _record_interrupts(monkeypatch)
    middleware = DeliveryGuardMiddleware()
    state = {"messages": [AIMessage(content="No new shipping updates.", id="ai-final")]}

    assert middleware.after_model(state, _runtime("cron")) is None
    assert payloads == []
