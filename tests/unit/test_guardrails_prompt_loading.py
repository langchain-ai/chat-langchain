"""Tests for guardrails prompt loading behavior."""

import importlib
import os
from contextlib import contextmanager

os.environ["USE_LOCAL_PROMPTS"] = "1"

import langsmith
from langsmith import run_helpers

from src.middleware import guardrails_middleware


def test_hub_prompt_rendering_disables_tracing(monkeypatch):
    contexts: list[dict[str, bool]] = []
    invoke_inputs: list[dict[str, list[object]]] = []

    class FakePrompt:
        metadata = {"lc_hub_commit_hash": "commit-123"}

        def invoke(self, prompt_input):
            invoke_inputs.append(prompt_input)
            return type(
                "PromptResult",
                (),
                {"messages": [type("Message", (), {"content": "hub prompt"})()]},
            )()

    class FakeClient:
        def pull_prompt(self, prompt_name):
            assert prompt_name == "public-chat-langchain-guardrails-test:production"
            return FakePrompt()

    @contextmanager
    def fake_tracing_context(**kwargs):
        contexts.append(kwargs)
        yield

    monkeypatch.delenv("USE_LOCAL_PROMPTS", raising=False)
    monkeypatch.setattr(langsmith, "Client", FakeClient)
    monkeypatch.setattr(run_helpers, "tracing_context", fake_tracing_context)

    try:
        loaded_module = importlib.reload(guardrails_middleware)
        loaded_prompt = loaded_module._GUARDRAILS_SYSTEM_PROMPT
        loaded_commit = loaded_module.guardrails_prompt_commit
    finally:
        monkeypatch.setenv("USE_LOCAL_PROMPTS", "1")
        importlib.reload(guardrails_middleware)

    assert loaded_prompt == "hub prompt"
    assert loaded_commit == "commit-123"
    assert contexts == [{"enabled": False}]
    assert invoke_inputs == [{"messages": []}]
