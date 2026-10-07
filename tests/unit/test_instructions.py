from pathlib import Path


def test_latest_message_takes_precedence_over_unanswered_history():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert "Answer the user's latest message" in instructions
    assert "earlier unanswered messages are context only" in instructions
    assert "address them only if the latest message refers to them" in instructions


def test_yes_no_answers_require_retrieved_evidence():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert "For a yes/no question, first check whether a retrieved" in instructions
    assert "the documentation does not state the claim" in instructions
    assert "require a quoted source from the retrieved content" in instructions
