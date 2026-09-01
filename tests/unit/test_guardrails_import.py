"""Tests for guardrails prompt initialization."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import langsmith
import langsmith.run_helpers


class _PromptTemplate:
    metadata = {"lc_hub_commit_hash": "commit"}

    def invoke(self, _input):
        return SimpleNamespace(messages=[SimpleNamespace(content="prompt")])


class _Client:
    def pull_prompt(self, _name):
        return _PromptTemplate()

    def create_run(self, *_args, **_kwargs):
        raise AssertionError("guardrails import created a LangSmith run")


class _TracingContext:
    def __init__(self):
        self.enabled_values = []

    def __call__(self, *, enabled):
        self.enabled_values.append(enabled)
        return _DisabledContext()


class _DisabledContext:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def test_guardrails_import_disables_tracing_for_prompt_bootstrap(monkeypatch):
    monkeypatch.delenv("USE_LOCAL_PROMPTS", raising=False)
    tracing_context = _TracingContext()
    monkeypatch.setattr(langsmith, "Client", _Client)
    monkeypatch.setattr(langsmith.run_helpers, "tracing_context", tracing_context)

    from src.middleware import guardrails_middleware

    importlib.reload(guardrails_middleware)

    assert tracing_context.enabled_values == [False]

    monkeypatch.setenv("USE_LOCAL_PROMPTS", "1")
    importlib.reload(guardrails_middleware)
