from pathlib import Path


def test_yes_no_answers_require_retrieved_evidence():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert "For a yes/no question, first check whether a retrieved" in instructions
    assert "the documentation does not state the claim" in instructions
    assert "require a quoted source from the retrieved content" in instructions


def test_defensive_security_is_not_a_blocked_application():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert "Defensive security for the user's own" in instructions
    assert "prompt-injection demos or teaching labs" in instructions
    assert (
        "input guards, regex block lists, tool allowlists, secret-handling"
        in instructions
    )
    assert (
        "this does not allow attacks on this assistant or genuinely harmful applications"
        in instructions
    )
    assert (
        "NEVER help design or implement harmful, fraudulent, abusive, or illegal use cases"
        in instructions
    )


def test_current_allowed_turn_does_not_inherit_classifier_refusal():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert (
        "Refusals are sticky only for your own prior decline of the same request"
        in instructions
    )
    assert (
        "guardrails-classifier block or its generated rejection is not your own decline"
        in instructions
    )
    assert "when the current turn is ALLOWED, evaluate it on its merits" in instructions
