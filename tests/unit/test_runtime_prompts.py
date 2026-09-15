"""Tests for the prompts loaded by the managed agent runtime."""

from __future__ import annotations

import importlib
import sys
import types

from src.prompts.docs_agent_prompt import docs_agent_prompt


def test_mda_runtime_uses_docs_agent_prompt(monkeypatch):
    runtime = types.ModuleType("managed_deepagents.runtime")
    runtime.compile_managed_agent = lambda *args, **kwargs: None
    monkeypatch.setitem(
        sys.modules, "managed_deepagents", types.ModuleType("managed_deepagents")
    )
    monkeypatch.setitem(sys.modules, "managed_deepagents.runtime", runtime)
    monkeypatch.setitem(
        sys.modules, "_mda_connectors", types.SimpleNamespace(connectors={})
    )
    monkeypatch.setitem(sys.modules, "agent", types.SimpleNamespace(agent=object()))
    monkeypatch.setitem(
        sys.modules, "identity", types.SimpleNamespace(identity=object())
    )

    module = importlib.import_module("_mda_entry")
    importlib.reload(module)

    assert module._system_prompt == docs_agent_prompt
    for rule in (
        "You CANNOT open, create, file, or submit support tickets",
        "A follow-up question inside an ongoing conversation is NOT a clarification",
        "Support articles could not be consulted, so this answer is based on official documentation only.",
        "Copy citation URLs verbatim from this turn's documentation tool results.",
    ):
        assert rule in module._system_prompt
