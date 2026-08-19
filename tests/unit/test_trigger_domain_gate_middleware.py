"""Tests for the trigger domain gate and its wiring-time validation."""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

os.environ["USE_LOCAL_PROMPTS"] = "1"

from src.middleware.trigger_domain_gate_middleware import (
    TriggerContractError,
    TriggerDomainGateMiddleware,
    declares_email_contract,
    is_email_trigger,
    is_trigger_payload,
    parse_trigger_name,
    validate_trigger_wiring,
)

AGENT_DOMAIN = (
    "Competitor intelligence for the default company: research tracked competitors "
    "and maintain the competitor wiki and battlecards."
)

FIGMA_TRIGGER = """ACTION: gmail email received
EMAIL: jeeyoon@langchain.dev
GMAIL THREAD ID: 1a016060df97e10f

MESSAGE UPDATES:
Figma <no-reply@figma.com> sent this email:
Subject: 3 new comments in LangSmith Dashboard Redesign - Agent Lifecycle
"""

PAYROLL_TRIGGER = """ACTION: gmail email received
EMAIL: jeeyoon@langchain.dev

MESSAGE UPDATES:
LangChain Payroll <payroll@langchain.dev> sent this email:
Subject: Recent Corrections Made to your Account (No Action Required)
Body: an issue in payroll processing caused certain 401(k) contributions to be
under-withheld.
"""

COMPETITOR_TRIGGER = """ACTION: gmail email received
EMAIL: jeeyoon@langchain.dev

MESSAGE UPDATES:
Subject: Competitor X launches usage-based pricing tier
"""

SLACK_AND_SCHEDULED_ONLY_INSTRUCTIONS = """# Competitor Intelligence Slack Bot

## Primary Surface - Slack
Answer competitor questions asked in Slack threads.

## Scheduled Refreshes
Refresh the wiki on the weekly schedule. Do not send Slack messages from
scheduled runs.
"""

INSTRUCTIONS_WITH_EMAIL_CONTRACT = (
    SLACK_AND_SCHEDULED_ONLY_INSTRUCTIONS
    + """
## Email received
Decide whether the email concerns a tracked competitor. If it does not, the run
is a no-op: emit a single line noting the trigger is outside competitor-
intelligence scope and never summarize or reply to the email.
"""
)


def _classifier(decision: str, explanation: str = "test"):
    async def classify(payload: str):
        return {"decision": decision, "explanation": explanation}

    return classify


def _gate(decision: str = "BLOCK", **kwargs) -> TriggerDomainGateMiddleware:
    return TriggerDomainGateMiddleware(
        agent_domain=AGENT_DOMAIN,
        classifier=_classifier(decision),
        **kwargs,
    )


def test_parse_trigger_name_and_email_detection():
    assert parse_trigger_name(FIGMA_TRIGGER) == "gmail email received"
    assert parse_trigger_name("what do we know about Competitor X?") is None
    assert is_email_trigger("gmail email received")
    assert not is_email_trigger("slack message posted")


def test_is_trigger_payload_uses_header_or_metadata():
    assert is_trigger_payload(FIGMA_TRIGGER)
    assert is_trigger_payload("weekly refresh", {"source": "trigger"})
    assert not is_trigger_payload("weekly refresh", {"source": "chat"})


@pytest.mark.parametrize("payload", [FIGMA_TRIGGER, PAYROLL_TRIGGER])
def test_off_domain_email_trigger_short_circuits_with_scope_note(payload):
    gate = _gate("BLOCK")
    state = {"messages": [HumanMessage(content=payload, id="h1")]}

    update = asyncio.run(gate.abefore_agent(state, runtime=SimpleNamespace()))

    assert update is not None
    assert update["jump_to"] == "end"
    assert update["off_domain_trigger"] is True
    reply = update["messages"][0]
    assert isinstance(reply, AIMessage)
    assert len(reply.content.splitlines()) == 1
    assert "outside my scope" in reply.content
    # The decline must not restate the payload it declined to handle.
    assert "401(k)" not in reply.content
    assert "Figma" not in reply.content


def test_in_domain_email_trigger_reaches_the_agent():
    gate = _gate("ALLOW")
    state = {"messages": [HumanMessage(content=COMPETITOR_TRIGGER, id="h1")]}

    assert asyncio.run(gate.abefore_agent(state, runtime=SimpleNamespace())) is None


def test_interactive_chat_turn_is_not_gated():
    gate = _gate("BLOCK")
    state = {"messages": [HumanMessage(content="hey, what changed this week?", id="h1")]}

    assert asyncio.run(gate.abefore_agent(state, runtime=SimpleNamespace())) is None


def test_classifier_failure_fails_closed_on_trigger_runs():
    async def failing_classifier(payload: str):
        raise RuntimeError("classifier unavailable")

    gate = TriggerDomainGateMiddleware(
        agent_domain=AGENT_DOMAIN, classifier=failing_classifier
    )
    state = {"messages": [HumanMessage(content=PAYROLL_TRIGGER, id="h1")]}

    update = asyncio.run(gate.abefore_agent(state, runtime=SimpleNamespace()))

    assert update["jump_to"] == "end"


def test_block_off_domain_false_only_observes():
    gate = _gate("BLOCK", block_off_domain=False)
    state = {"messages": [HumanMessage(content=FIGMA_TRIGGER, id="h1")]}

    assert asyncio.run(gate.abefore_agent(state, runtime=SimpleNamespace())) is None


def test_gate_requires_a_declared_domain():
    with pytest.raises(TriggerContractError):
        TriggerDomainGateMiddleware(agent_domain="  ", classifier=_classifier("ALLOW"))


def test_validate_trigger_wiring_flags_email_trigger_without_contract():
    problems = validate_trigger_wiring(
        ["gmail email received", "slack message posted"],
        SLACK_AND_SCHEDULED_ONLY_INSTRUCTIONS,
    )

    assert len(problems) == 1
    assert "gmail email received" in problems[0]

    with pytest.raises(TriggerContractError):
        validate_trigger_wiring(
            ["gmail email received"],
            SLACK_AND_SCHEDULED_ONLY_INSTRUCTIONS,
            strict=True,
        )


def test_validate_trigger_wiring_passes_when_contract_declared():
    assert declares_email_contract(INSTRUCTIONS_WITH_EMAIL_CONTRACT)
    assert not declares_email_contract(SLACK_AND_SCHEDULED_ONLY_INSTRUCTIONS)
    assert (
        validate_trigger_wiring(
            ["gmail email received"], INSTRUCTIONS_WITH_EMAIL_CONTRACT, strict=True
        )
        == []
    )
