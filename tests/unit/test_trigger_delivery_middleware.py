"""Tests for the scheduled-run terminal-delivery guard."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

os.environ["USE_LOCAL_PROMPTS"] = "1"

from src.middleware.trigger_delivery_middleware import (
    TriggerDeliveryGuardMiddleware,
)


def _runtime(provider: str | None):
    if provider is None:
        return SimpleNamespace(configurable={})
    return SimpleNamespace(identity={"source": {"provider": provider}})


def _tool_call(name: str):
    return {"name": name, "args": {}, "id": f"call_{name}", "type": "tool_call"}


def test_reprompts_when_scheduled_run_ends_without_delivery():
    middleware = TriggerDeliveryGuardMiddleware()
    state = {
        "messages": [
            HumanMessage(content="scan gmail and build a status page"),
            AIMessage(content="", tool_calls=[_tool_call("write_file")]),
            AIMessage(content="Done — see /tmp/status.html"),
        ]
    }

    update = middleware.after_model(state, runtime=_runtime("schedule"))

    assert update is not None
    assert update["jump_to"] == "model"
    assert update["delivery_reprompts"] == 1
    assert "/tmp/" in update["messages"][0].content


def test_noop_when_delivery_tool_was_called():
    middleware = TriggerDeliveryGuardMiddleware()
    state = {
        "messages": [
            AIMessage(content="", tool_calls=[_tool_call("gmail_send_email")]),
            AIMessage(content="Sent the digest."),
        ]
    }

    assert middleware.after_model(state, runtime=_runtime("schedule")) is None


def test_noop_for_interactive_runs():
    middleware = TriggerDeliveryGuardMiddleware()
    state = {"messages": [AIMessage(content="Done — see /tmp/status.html")]}

    assert middleware.after_model(state, runtime=_runtime("http")) is None
    assert middleware.after_model(state, runtime=_runtime(None)) is None


def test_noop_while_the_model_is_still_calling_tools():
    middleware = TriggerDeliveryGuardMiddleware()
    state = {"messages": [AIMessage(content="", tool_calls=[_tool_call("write_file")])]}

    assert middleware.after_model(state, runtime=_runtime("schedule")) is None


def test_reprompt_budget_is_bounded():
    middleware = TriggerDeliveryGuardMiddleware()
    state = {
        "messages": [AIMessage(content="Done — see /tmp/status.html")],
        "delivery_reprompts": 1,
    }

    assert middleware.after_model(state, runtime=_runtime("schedule")) is None


@pytest.mark.asyncio
async def test_delivery_contract_is_added_to_scheduled_system_message():
    middleware = TriggerDeliveryGuardMiddleware()
    seen: list[SystemMessage | None] = []

    async def handler(request):
        seen.append(request.system_message)
        return AIMessage(content="ok")

    request = SimpleNamespace(
        runtime=_runtime("schedule"),
        system_message=SystemMessage(content="base prompt"),
    )
    request.override = lambda **kwargs: SimpleNamespace(**{**vars(request), **kwargs})

    await middleware.awrap_model_call(request, handler)

    assert seen[0] is not None
    assert seen[0].content.startswith("base prompt")
    assert "scheduled_run_delivery" in seen[0].content


@pytest.mark.asyncio
async def test_delivery_contract_is_absent_for_interactive_runs():
    middleware = TriggerDeliveryGuardMiddleware()
    seen: list[SystemMessage | None] = []

    async def handler(request):
        seen.append(request.system_message)
        return AIMessage(content="ok")

    request = SimpleNamespace(
        runtime=_runtime("http"),
        system_message=SystemMessage(content="base prompt"),
    )

    await middleware.awrap_model_call(request, handler)

    assert seen[0].content == "base prompt"
