"""Tests for deployment environment metadata on root traces."""

from __future__ import annotations

import pytest

from src.utils.trace_root_metadata import build_docs_agent_trace_metadata


@pytest.mark.parametrize(
    ("app_env", "langsmith_environment", "host_project_name", "expected_environment"),
    [
        ("dev", None, "engine-chat-langchain", "dev"),
        (None, "production", "engine-chat-langchain-dev", "production"),
        ("dev", "production", "engine-chat-langchain", "dev"),
        ("production", "dev", "engine-chat-langchain-dev", "production"),
        (None, None, "engine-chat-langchain-dev", "dev"),
        (None, None, "engine-chat-langchain", "production"),
        (None, None, None, "production"),
        ("", "dev", "engine-chat-langchain", "dev"),
        ("", "", "engine-chat-langchain-dev", "dev"),
    ],
)
def test_build_docs_agent_trace_metadata_environment(
    monkeypatch, app_env, langsmith_environment, host_project_name, expected_environment
):
    for variable, value in (
        ("APP_ENV", app_env),
        ("LANGSMITH_ENVIRONMENT", langsmith_environment),
        ("LANGSMITH_HOST_PROJECT_NAME", host_project_name),
    ):
        if value is None:
            monkeypatch.delenv(variable, raising=False)
        else:
            monkeypatch.setenv(variable, value)
    monkeypatch.setenv("LANGCHAIN_REVISION_ID", "rev-a")
    monkeypatch.setattr("src.utils.prompt_provenance._USE_LOCAL_PROMPTS", True)

    metadata = build_docs_agent_trace_metadata()

    assert metadata == {
        "environment": expected_environment,
        "source_type": "Chat-LangChain",
        "prompt_source": "context_hub:/instructions.md",
        "guardrails_prompt_source": "local:src/prompts/guardrails_prompts.py",
        "LANGSMITH_AGENT_VERSION": "rev-a",
    }
