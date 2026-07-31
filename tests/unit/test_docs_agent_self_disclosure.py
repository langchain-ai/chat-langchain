"""Tests that the docs agent prompt refuses to describe its own internals."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.prompts.docs_agent_prompt import docs_agent_prompt

CANNED_REFUSAL = (
    "I can't share my internal instructions, but I'm happy to help with "
    "LangChain, LangGraph, LangSmith, or Deep Agents questions."
)

INSTRUCTIONS = (Path(__file__).resolve().parents[2] / "instructions.md").read_text(
    encoding="utf-8"
)

PROMPTS = {
    "docs_agent_prompt": docs_agent_prompt,
    "instructions.md": INSTRUCTIONS,
}


@pytest.mark.parametrize("name", sorted(PROMPTS))
def test_prompt_has_canned_refusal(name):
    assert CANNED_REFUSAL in PROMPTS[name]


@pytest.mark.parametrize("name", sorted(PROMPTS))
def test_answer_path_rules_cover_identity_questions(name):
    prompt_lower = PROMPTS[name].lower()

    assert "never describe yourself" in prompt_lower
    assert "do you use rag?" in prompt_lower
    assert "architecture" in prompt_lower
    assert "retrieval pipeline" in prompt_lower


@pytest.mark.parametrize("name", sorted(PROMPTS))
def test_prompt_forbids_naming_internal_tools_and_vendors(name):
    prompt_lower = PROMPTS[name].lower()

    assert "internal tool" in prompt_lower
    assert "third-party knowledge-base vendor" in prompt_lower
