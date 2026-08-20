"""Tests for the deliverables / sandbox capability boundary in the system prompt.

The agent runs with deep-agent filesystem tools whose sandbox is ephemeral and
not reachable by the user. Without an explicit boundary the agent answers
"page"/"dashboard"/"report" requests by writing a file and naming its `/tmp/`
path as the deliverable, which surfaces nothing to the user.
"""

from pathlib import Path

import pytest
from langsmith import testing as t

from src.prompts.docs_agent_prompt import docs_agent_prompt

PROMPT_LOWER = docs_agent_prompt.lower()

INSTRUCTIONS_MD = Path(__file__).resolve().parents[2] / "instructions.md"


@pytest.mark.langsmith
def test_prompt_declares_sandbox_is_not_user_reachable():
    """Prompt must say the filesystem sandbox is ephemeral and not user-reachable."""
    t.log_inputs({"check": "sandbox_boundary"})

    has_ephemeral = "ephemeral" in PROMPT_LOWER and "sandbox" in PROMPT_LOWER
    has_not_reachable = "cannot browse it" in PROMPT_LOWER
    has_write_is_not_delivery = "never by itself a delivered artifact" in PROMPT_LOWER

    t.log_outputs({
        "has_ephemeral": has_ephemeral,
        "has_not_reachable": has_not_reachable,
        "has_write_is_not_delivery": has_write_is_not_delivery,
    })
    t.log_reference_outputs({"all_present": True})

    assert has_ephemeral and has_not_reachable and has_write_is_not_delivery, (
        "System prompt must state that the filesystem tools write to an "
        "ephemeral per-run sandbox the user cannot browse, and that writing a "
        "file is never by itself a delivered artifact."
    )


@pytest.mark.langsmith
def test_prompt_forbids_citing_sandbox_path_as_deliverable():
    """Prompt must forbid handing the user a sandbox path as the deliverable."""
    t.log_inputs({"check": "no_sandbox_path_as_deliverable"})

    forbids_path = "sandbox path" in PROMPT_LOWER and "deliverable" in PROMPT_LOWER

    t.log_outputs({"forbids_path": forbids_path})
    t.log_reference_outputs({"forbids_path": True})

    assert forbids_path, (
        "System prompt must forbid presenting a sandbox path (e.g. "
        "`/tmp/status.html`) to the user as the deliverable."
    )


@pytest.mark.langsmith
def test_prompt_requires_delivery_or_explicit_declination():
    """Page/dashboard/report requests must route to a real channel or a declination."""
    t.log_inputs({"check": "delivery_contract"})

    artifact_words = ["page", "dashboard", "live view", "digest", "report"]
    missing_artifact_words = [w for w in artifact_words if w not in PROMPT_LOWER]

    channels = ["slack", "email", "inline"]
    missing_channels = [c for c in channels if c not in PROMPT_LOWER]

    has_declination = (
        "not something you can do" in PROMPT_LOWER
        and "publishing or hosting" in PROMPT_LOWER
    )

    t.log_outputs({
        "missing_artifact_words": missing_artifact_words,
        "missing_channels": missing_channels,
        "has_declination": has_declination,
    })
    t.log_reference_outputs({"missing": [], "has_declination": True})

    assert not missing_artifact_words, (
        f"Prompt must name the request shapes that require delivery. "
        f"Missing: {missing_artifact_words}"
    )
    assert not missing_channels, (
        f"Prompt must name the real delivery channels to use. "
        f"Missing: {missing_channels}"
    )
    assert has_declination, (
        "Prompt must tell the agent to explicitly decline hosting/publishing a "
        "live page when it has no such capability, instead of silently writing a file."
    )


@pytest.mark.langsmith
def test_prompt_forbids_calling_static_files_live():
    """Prompt must forbid describing a statically written file as live-updating."""
    t.log_inputs({"check": "no_live_claims"})

    forbidden_claims = ["live-updating", "auto-refreshing"]
    missing = [c for c in forbidden_claims if c not in PROMPT_LOWER]

    t.log_outputs({"missing_forbidden_claims": missing})
    t.log_reference_outputs({"missing_forbidden_claims": []})

    assert not missing, (
        f"Prompt must explicitly forbid describing a statically written file as "
        f"live/live-updating/auto-refreshing. Missing terms: {missing}"
    )


@pytest.mark.langsmith
def test_instructions_md_mirrors_the_delivery_contract():
    """instructions.md (the deployed prompt) must carry the same delivery contract."""
    t.log_inputs({"check": "instructions_md_mirror"})

    instructions = INSTRUCTIONS_MD.read_text()
    has_section = "## Deliverables and Capability Boundaries" in instructions

    t.log_outputs({"has_section": has_section})
    t.log_reference_outputs({"has_section": True})

    assert has_section, (
        "instructions.md is the prompt the deployed agent actually runs. It must "
        "stay in sync with src/prompts/docs_agent_prompt.py and contain the "
        "'Deliverables and Capability Boundaries' section."
    )
