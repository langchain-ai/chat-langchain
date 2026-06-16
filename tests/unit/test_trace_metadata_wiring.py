"""Tests proving trace metadata is wired into run config and model spans."""

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import patch


def test_build_trace_metadata_includes_filtering_and_cost_keys():
    from src.utils.trace_metadata import build_trace_metadata

    metadata = build_trace_metadata(
        user_id="user@example.com",
        thread_id="t-1",
        environment="prod",
        graph_id="docs_agent",
        prompt_source="hub:foo:production",
        prompt_commit="abc",
        guardrails_prompt_source="hub:foo-guardrails:production",
        guardrails_prompt_commit="def",
    )

    assert metadata["user_id"] == "user@example.com"
    assert metadata["thread_id"] == "t-1"
    assert metadata["environment"] == "prod"
    assert metadata["graph_id"] == "docs_agent"
    assert metadata["prompt_source"] == "hub:foo:production"
    assert metadata["prompt_commit"] == "abc"
    assert metadata["guardrails_prompt_source"] == "hub:foo-guardrails:production"
    assert metadata["guardrails_prompt_commit"] == "def"


def test_enrich_run_metadata_stamps_trace_metadata_keys(monkeypatch):
    """Root docs_agent runs receive thread_id, environment, and prompt provenance."""
    from src.api import auth
    from src.api.auth import _reset_rate_limit_for_tests

    _reset_rate_limit_for_tests()
    monkeypatch.setenv("LANGSMITH_ENV", "prod")
    ctx = SimpleNamespace(user={"identity": "user@example.com", "client_ip": "1.2.3.4"})
    value = {
        "assistant_id": "docs_agent",
        "thread_id": "11111111-1111-1111-1111-111111111111",
        "kwargs": {
            "input": {"messages": [{"role": "user", "content": "hi"}]},
            "config": {},
        },
    }

    with patch.object(
        auth,
        "get_prompt_provenance",
        return_value={
            "prompt_source": "hub:p:production",
            "prompt_commit": "deadbeef",
            "guardrails_prompt_source": "hub:g:production",
            "guardrails_prompt_commit": "cafebabe",
        },
    ):
        asyncio.run(auth.enrich_run_metadata(ctx, value))

    metadata = value["metadata"]
    assert metadata["user_id"] == "user@example.com"
    assert metadata["source_type"] == "Chat-LangChain"
    assert metadata["thread_id"] == "11111111-1111-1111-1111-111111111111"
    assert metadata["environment"] == "prod"
    assert metadata["graph_id"] == "docs_agent"
    assert metadata["prompt_source"] == "hub:p:production"
    assert metadata["prompt_commit"] == "deadbeef"
    assert metadata["guardrails_prompt_source"] == "hub:g:production"
    assert metadata["guardrails_prompt_commit"] == "cafebabe"


def test_default_model_carries_ls_provider_and_ls_model_name():
    """Chat models bind LangSmith provider/model metadata for cost accounting."""
    os.environ.setdefault("OPENAI_API_KEY", "sk-test")
    os.environ.setdefault("ANTHROPIC_API_KEY", "sk-test")
    os.environ.setdefault("GOOGLE_API_KEY", "sk-test")

    from src.agent.config import (
        DEFAULT_MODEL,
        _ls_metadata_for,
        default_model,
        init_retry_fallback_model,
        summarization_model,
    )

    default_meta = default_model.config.get("metadata", {})
    assert default_meta.get("ls_provider")
    assert default_meta.get("ls_model_name") == DEFAULT_MODEL.id.split(":", 1)[1]

    summarization_meta = summarization_model.config.get("metadata", {})
    assert summarization_meta.get("ls_provider")
    assert summarization_meta.get("ls_model_name")

    fallback = init_retry_fallback_model(DEFAULT_MODEL.id)
    fallback_meta = fallback.config.get("metadata", {})
    assert fallback_meta.get("ls_provider")
    assert fallback_meta.get("ls_model_name")

    helper_meta = _ls_metadata_for("openai:gpt-4o-mini")
    assert helper_meta == {
        "ls_model_type": "chat",
        "ls_provider": "openai",
        "ls_model_name": "gpt-4o-mini",
    }
