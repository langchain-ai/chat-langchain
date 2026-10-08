from pathlib import Path


def test_yes_no_answers_require_retrieved_evidence():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert "For a yes/no question, first check whether a retrieved" in instructions
    assert "the documentation does not state the claim" in instructions
    assert "require a quoted source from the retrieved content" in instructions


def test_scope_clearance_and_refusals_are_current_request_specific():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert "current-turn guardrails ALLOWED verdict as scope clearance" in instructions
    assert (
        "ask exactly one clarifying question instead of a scope refusal" in instructions
    )
    assert "classifier-BLOCKED or unambiguously off-domain turns" in instructions
    assert "Refusals are sticky only for the specific declined request" in instructions
    assert "same declined content through reframing or pushback" in instructions
    assert "Judge new or different questions on their own merits" in instructions
