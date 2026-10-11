"""Tests for code-review grounding in the deployed prompt and Hub mirror."""

from pathlib import Path

import pytest

from src.prompts.docs_agent_prompt import docs_agent_prompt

INSTRUCTIONS_PATH = Path(__file__).resolve().parents[2] / "instructions.md"


def _review_section(prompt):
    return prompt.split("## Reviewing user code\n", 1)[1].split("\n## ", 1)[0]


@pytest.fixture(params=["managed", "hub_mirror"])
def review_instructions(request):
    prompt = (
        INSTRUCTIONS_PATH.read_text()
        if request.param == "managed"
        else docs_agent_prompt
    )
    return _review_section(prompt)


def test_managed_review_guidance_matches_hub_mirror():
    assert _review_section(INSTRUCTIONS_PATH.read_text()) == _review_section(
        docs_agent_prompt
    )


def test_review_uses_latest_actual_submission(review_instructions):
    assert "latest actual user-submitted code" in review_instructions
    assert "not superseded snippets or assistant rewrites" in review_instructions
    assert (
        "Injected research/citation retry instructions are not new code submissions"
        in review_instructions
    )


def test_review_quotes_submission_without_exposing_secrets(review_instructions):
    assert (
        "Ground each reported defect in that submission and the docs"
        in review_instructions
    )
    assert "quote the exact relevant line, redacting secrets" in review_instructions


@pytest.mark.parametrize("omission", ["imports", "node registrations", "edges"])
def test_review_checks_latest_code_before_claiming_omissions(
    review_instructions, omission
):
    assert (
        "Verify omissions against the entire latest submission" in review_instructions
    )
    assert omission in review_instructions
    assert "Report an omission only if absent" in review_instructions
    assert "do not invent a line to quote for an omission" in review_instructions


def test_review_acknowledges_fixed_defects(review_instructions):
    assert "Acknowledge corrected earlier defects" in review_instructions
    assert "do not repeat them as current problems" in review_instructions
    assert (
        "Keep the bold opening verdict and existing answer format"
        in review_instructions
    )


def test_review_guidance_stays_concise(review_instructions):
    assert len(review_instructions.strip().splitlines()) < 10
