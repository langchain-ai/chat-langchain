"""Tests for guardrails prompt loading."""

import os

os.environ["USE_LOCAL_PROMPTS"] = "1"

from src.middleware import guardrails_middleware as guardrails_module


def test_guardrails_prompt_is_loaded_at_module_import():
    """The guardrails prompt should be available when the module loads."""
    assert guardrails_module._GUARDRAILS_SYSTEM_PROMPT.startswith(
        "You are a lenient content filter"
    )
