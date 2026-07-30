"""Tests that the guardrails hub prompt bootstrap emits no LangSmith run."""

import os

from langsmith.run_helpers import get_tracing_context

os.environ["USE_LOCAL_PROMPTS"] = "1"

from src.middleware import guardrails_middleware as guardrails_module


class _FakeRendered:
    def __init__(self, content):
        self.messages = [type("Msg", (), {"content": content})()]


class _FakeTemplate:
    def __init__(self):
        self.tracing_enabled_during_invoke = "unset"

    def invoke(self, variables):
        assert variables == {"messages": []}
        self.tracing_enabled_during_invoke = get_tracing_context()["enabled"]
        return _FakeRendered("hub guardrails prompt")


def test_render_hub_prompt_disables_tracing():
    template = _FakeTemplate()

    assert guardrails_module._render_hub_prompt(template) == "hub guardrails prompt"
    assert template.tracing_enabled_during_invoke is False


def test_render_hub_prompt_restores_tracing_context():
    before = get_tracing_context()["enabled"]

    guardrails_module._render_hub_prompt(_FakeTemplate())

    assert get_tracing_context()["enabled"] == before
