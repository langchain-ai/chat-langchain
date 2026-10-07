from pathlib import Path


def test_latest_user_message_takes_priority_over_interrupted_turns():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert (
        "Always answer the most recent user message, including its constraints"
        in instructions
    )
    assert (
        "Unanswered earlier user messages belong to stopped or failed turns"
        in instructions
    )
    assert "unless the latest message asks for them" in instructions


def test_yes_no_answers_require_retrieved_evidence():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert "For a yes/no question, first check whether a retrieved" in instructions
    assert "the documentation does not state the claim" in instructions
    assert "require a quoted source from the retrieved content" in instructions
