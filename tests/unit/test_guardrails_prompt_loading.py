"""Tests for guardrails prompt loading."""

import importlib
from types import SimpleNamespace

import langsmith

from src.middleware import guardrails_middleware


def test_import_uses_template_text_without_invoking_prompt(monkeypatch):
    """Importing guardrails does not invoke the pulled prompt runnable."""
    monkeypatch.delenv("USE_LOCAL_PROMPTS", raising=False)

    class FakePromptTemplate:
        messages = [SimpleNamespace(prompt=SimpleNamespace(template="hub prompt"))]
        metadata = {}

        def invoke(self, input):  # noqa: ARG002
            raise AssertionError("prompt invocation should not occur during import")

    class FakeClient:
        def pull_prompt(self, prompt_name):  # noqa: ARG002
            return FakePromptTemplate()

    monkeypatch.setattr(langsmith, "Client", FakeClient)
    reloaded = importlib.reload(guardrails_middleware)

    assert reloaded._GUARDRAILS_SYSTEM_PROMPT == "hub prompt"

    monkeypatch.setenv("USE_LOCAL_PROMPTS", "1")
    importlib.reload(guardrails_middleware)
